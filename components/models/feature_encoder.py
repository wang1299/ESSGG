import json
import math
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch_geometric.data import Data, HeteroData
from torchvision.models import resnet18, ResNet18_Weights

from components.models.graph_encoder import NodeEdgeHGTEncoder


class FeatureEncoder(nn.Module):
    """
    Encodes the multimodal agent state including:
    - RGB image via ResNet18
    - Last action via embedding
    - Occupancy map via CNN
    - Local and global scene graphs via LSTM **or** Transformer (configurable)
    Combines all features into a single state vector.
    """

    def __init__(
        self,
        num_actions,
        rgb_dim=512,
        action_dim=32,
        map_dim=64,
        sg_dim=256,
        obj_embedding_dim=128,
        max_object_types=1000,
        rel_embedding_dim=64,
        max_relation_types=50,
        use_transformer=False,
        mapping_path=None,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.rgb_encoder = ResNetFeatureExtractor(rgb_dim)
        self.action_emb = ActionEmbedding(num_actions, action_dim)

        self.sg_dim = sg_dim
        self.use_transformer = use_transformer

        SGEncoderClass = SceneGraphTransformerEncoder if use_transformer else SceneGraphLSTMEncoder

        self.lssg_encoder = SGEncoderClass(input_dim=sg_dim, hidden_dim=sg_dim)
        self.gssg_encoder = SGEncoderClass(input_dim=sg_dim, hidden_dim=sg_dim)
        self.node_att_vector = nn.Parameter(torch.randn(int(sg_dim / 2)))
        self.edge_att_vector = nn.Parameter(torch.randn(int(sg_dim / 2)))

        self.object_to_idx = {}
        self.relation_to_idx = {}  # New mapping for relations

        self.max_object_types = max_object_types
        self.max_relation_types = max_relation_types
        self.obj_type_embedding = nn.Embedding(max_object_types, obj_embedding_dim)
        self.rel_type_embedding = nn.Embedding(max_relation_types, rel_embedding_dim)  # New embedding layer

        self.mapping_path = mapping_path
        if mapping_path and os.path.exists(os.path.join(mapping_path, "object_types.json")):
            self.load_mappings(mapping_path)

        relation_types = list(self.relation_to_idx.keys())
        graph_encoder_in_channels = 4 + obj_embedding_dim  # Node features: visibility + pos (x, y, z) + obj_embedding
        self.graph_feature_extractor = NodeEdgeHGTEncoder(
            in_channels=graph_encoder_in_channels,
            edge_in_channels=rel_embedding_dim,
            hidden_channels=128,
            out_channels=int(sg_dim / 2),
            relation_types=relation_types,
        )

        self.object_count = len(self.object_to_idx)
        self.relation_count = len(self.relation_to_idx)

    @staticmethod
    def preprocess_rgb(rgb_list):
        """
        Accepts: list of np.ndarray, PIL.Image, or torch.Tensor
        Returns: FloatTensor [N, 3, H, W]
        """
        transform = T.Compose(
            [
                T.ToTensor(),  # Converts np.ndarray/PIL → Tensor [0,1]
                T.Resize((224, 224)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        processed = []
        for rgb in rgb_list:
            if rgb is None or (isinstance(rgb, int) and rgb == 0):
                # Padding
                processed.append(torch.zeros(3, 224, 224))
            elif isinstance(rgb, torch.Tensor):
                img = rgb
                if img.ndim == 3:
                    processed.append(transform(img))
                else:
                    processed.append(img)
            else:
                if isinstance(rgb, np.ndarray):
                    rgb = np.ascontiguousarray(rgb)
                elif hasattr(rgb, "copy"):  # e.g. PIL Image
                    rgb = np.array(rgb).copy()
                processed.append(transform(rgb))
        return torch.stack(processed)  # [N, 3, H, W]

    def create_hgt_data(self, sg, device):
        # Early exit for empty graphs
        if sg is None or (isinstance(sg, int) and sg == 0) or not sg.nodes:
            return None

        data = HeteroData()

        # Mapping from node_id to index
        node_id_map = {node_id: i for i, node_id in enumerate(sg.nodes)}

        # --- Node Features: position (3), visibility (1), object embedding (d) ---
        node_positions = []
        object_type_indices = []
        visibilities = []
        for node in sg.nodes.values():
            node_positions.append(node.position)
            obj_type_idx = self.object_to_idx.setdefault(node.name, len(self.object_to_idx))
            object_type_indices.append(obj_type_idx)
            visibilities.append(getattr(node, "visibility", 1.0))
        obj_indices_tensor = torch.tensor(object_type_indices, dtype=torch.long, device=device)
        obj_embeddings = self.obj_type_embedding(obj_indices_tensor)
        pos_tensor = torch.tensor(node_positions, dtype=torch.float32, device=device)
        vis_tensor = torch.tensor(visibilities, dtype=torch.float32, device=device).unsqueeze(1)
        x = torch.cat([pos_tensor, vis_tensor, obj_embeddings], dim=1)
        data["object"].x = x

        # --- Edge Index + Attr ---
        for rel_type in self.relation_to_idx:
            sources, targets, edge_attr_idx = [], [], []
            for edge in sg.edges:
                if edge.relation == rel_type:
                    if edge.source in node_id_map and edge.target in node_id_map:
                        sources.append(node_id_map[edge.source])
                        targets.append(node_id_map[edge.target])
                        idx = self.relation_to_idx[rel_type]
                        edge_attr_idx.append(idx)
            edge_type = ("object", rel_type, "object")
            if sources:
                data[edge_type].edge_index = torch.tensor([sources, targets], dtype=torch.long, device=device)
                edge_attr_tensor = self.rel_type_embedding(torch.tensor(edge_attr_idx, dtype=torch.long, device=device))
                data[edge_type].edge_attr = edge_attr_tensor
            else:
                data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                data[edge_type].edge_attr = torch.empty((0, self.rel_type_embedding.embedding_dim), device=device)
        return data

    def attention_pooling(self, features, att_vector):
        if features.size(0) == 0:
            return torch.zeros_like(att_vector)
        scores = torch.matmul(features, att_vector)
        weights = torch.softmax(scores, dim=0)
        pooled = torch.sum(weights.unsqueeze(-1) * features, dim=0)
        return pooled

    def get_graph_features(self, sg_list: list):
        device = next(self.parameters()).device
        data_list, valid_indices = [], []
        for i, sg in enumerate(sg_list):
            data = self.create_hgt_data(sg, device)
            if data is not None:
                data_list.append(data)
                valid_indices.append(i)

        if not data_list:
            pooled_features = torch.zeros((0, self.sg_dim), device=device)
            return pooled_features, torch.tensor([], dtype=torch.long, device=device)

        graph_embeds = []
        for d in data_list:
            node_out, edge_out = self.graph_feature_extractor(d)  # node_out: [num_nodes, d], edge_out: [num_edges, d]
            node_pooled = (
                self.attention_pooling(node_out, self.node_att_vector) if node_out.shape[0] > 0 else torch.zeros_like(self.node_att_vector)
            )
            edge_pooled = (
                self.attention_pooling(edge_out, self.edge_att_vector) if edge_out.shape[0] > 0 else torch.zeros_like(self.edge_att_vector)
            )
            graph_embeds.append(torch.cat([node_pooled, edge_pooled], dim=-1))  # [d_node+d_edge]
        pooled_features = torch.stack(graph_embeds, dim=0)  # [n_valid, d_node+d_edge]
        return pooled_features, torch.tensor(valid_indices, dtype=torch.long, device=device)

    def obs_to_dict(self, obs):
        """
        Converts an Observation or a list of Observations to a dict for feature extraction.
        Handles both single observations and temporal sequences.
        Returns a dict matching the input structure expected by forward_seq.
        """
        if isinstance(obs, list):
            # Sequence of Observations (single batch, T steps)
            rgb = [o.state[0] for o in obs]
            lssg = [o.state[1] for o in obs]
            gssg = [o.state[2] for o in obs]
            occ = [o.state[3] for o in obs]
            agent_pos = [o.info.get("agent_pos", None) for o in obs]
            return {
                "rgb": [rgb],  # [B=1, T]
                "lssg": [lssg],  # [B=1, T]
                "gssg": [gssg],  # [B=1, T]
                "occupancy": [occ],  # [B=1, T]
                "agent_pos": [agent_pos],  # [B=1, T]
            }
        else:
            # Single Observation (batch=1, T=1)
            rgb = obs.state[0]
            lssg = obs.state[1]
            gssg = obs.state[2]
            occ = obs.state[3]
            agent_pos = obs.info.get("agent_pos", None)
            return {"rgb": [[rgb]], "lssg": [[lssg]], "gssg": [[gssg]], "occupancy": [[occ]], "agent_pos": [[agent_pos]]}  # [B=1, T=1]

    def forward(self, obs, last_action, lssg_hidden=None, gssg_hidden=None):
        """
        Forward pass for a single observation or a sequence.
        Unifies preprocessing so that RL and IL use the same code path.
        obs: Observation or list of Observations
        last_action: int, list[int], or LongTensor [T] or [B,T]
        """
        # 1. Convert obs to batch_dict format
        batch_dict = self.obs_to_dict(obs) if not isinstance(obs, dict) else obs

        # 2. Normalize last_action to shape [B, T]
        device = next(self.parameters()).device
        if isinstance(last_action, int):
            last_action = torch.tensor([[last_action]], dtype=torch.long, device=device)
        elif isinstance(last_action, torch.Tensor):
            if last_action.ndim == 1:
                last_action = last_action.unsqueeze(0)  # [T] -> [1,T]
            elif last_action.ndim == 0:
                last_action = last_action.view(1, 1)
            last_action = last_action.to(device)
        elif isinstance(last_action, list):
            last_action = torch.tensor([last_action], dtype=torch.long, device=device)
        else:
            raise ValueError("last_action must be int, Tensor, or list of int.")

        # 3. Feature extraction via forward_seq (handles [B,T,...])
        return self.forward_seq(batch_dict, last_action, lssg_hidden=lssg_hidden, gssg_hidden=gssg_hidden)  # [B, T, D_total]

    def forward_seq(self, batch_dict, last_actions, pad_mask=None, lssg_hidden=None, gssg_hidden=None):
        """
        Preprocess and encode batch of sequences. Inputs: "raw" batch from seq_collate.
        batch_dict keys: 'occupancy', 'rgb', 'lssg', 'gssg', 'agent_pos'
        """
        device = next(self.parameters()).device
        B, T = len(batch_dict["rgb"]), len(batch_dict["rgb"][0])
        total_steps = B * T

        # --- 1. Process Standard Features (RGB, Occupancy, Action) ---
        rgb_flat = [im for seq in batch_dict["rgb"] for im in seq]

        rgb_tensor = self.preprocess_rgb(rgb_flat).to(device)
        act_flat = last_actions.view(-1).to(device)

        rgb_feat = self.rgb_encoder(rgb_tensor)
        act_feat = self.action_emb(act_flat)

        # --- 2. Process Graph Features with GAT ---
        lssg_flat = [sg for seq in batch_dict["lssg"] for sg in seq]
        gssg_flat = [sg for seq in batch_dict["gssg"] for sg in seq]

        # Get embeddings and the indices of non-empty graphs
        lssg_embeds, lssg_valid_indices = self.get_graph_features(lssg_flat)
        gssg_embeds, gssg_valid_indices = self.get_graph_features(gssg_flat)

        # --- 3. Re-align Graph Features with other features ---
        # Create placeholder tensors and fill them with the GAT outputs
        lssg_feat_full = torch.zeros(total_steps, self.sg_dim, device=device)
        gssg_feat_full = torch.zeros(total_steps, self.sg_dim, device=device)

        lssg_feat_full[lssg_valid_indices] = lssg_embeds
        gssg_feat_full[gssg_valid_indices] = gssg_embeds

        # --- 4. Process Graph Sequences ---
        lssg_seq = lssg_feat_full.view(B, T, -1)
        gssg_seq = gssg_feat_full.view(B, T, -1)

        if self.use_transformer:
            if "lssg_mask" in batch_dict:
                lssg_mask = torch.tensor(batch_dict["lssg_mask"], dtype=torch.bool, device=device)
                gssg_mask = torch.tensor(batch_dict["gssg_mask"], dtype=torch.bool, device=device)
                lssg_feat = self.lssg_encoder(lssg_seq, pad_mask=~lssg_mask)
                gssg_feat = self.gssg_encoder(gssg_seq, pad_mask=~gssg_mask)
            else:
                lssg_feat = self.lssg_encoder(lssg_seq, pad_mask=pad_mask)
                gssg_feat = self.gssg_encoder(gssg_seq, pad_mask=pad_mask)
        else:
            lssg_feat, lssg_hidden = self.lssg_encoder(lssg_seq, lssg_hidden)
            gssg_feat, gssg_hidden = self.gssg_encoder(gssg_seq, gssg_hidden)

        lssg_feat = lssg_feat.reshape(total_steps, -1)
        gssg_feat = gssg_feat.reshape(total_steps, -1)

        # --- 5. Concatenate all features ---
        feats = [act_feat, rgb_feat, lssg_feat, gssg_feat]

        out = torch.cat(feats, dim=-1)
        return out.view(B, T, -1), lssg_hidden, gssg_hidden

    def save_mappings(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "object_types.json"), "w") as f:
            json.dump(self.object_to_idx, f)
        with open(os.path.join(path, "relation_types.json"), "w") as f:
            json.dump(self.relation_to_idx, f)

    def load_mappings(self, path: str):
        object_types_file = os.path.join(path, "object_types.json")
        relation_types_file = os.path.join(path, "relation_types.json")
        if os.path.exists(object_types_file):
            with open(object_types_file, "r") as f:
                self.object_to_idx = json.load(f)
                self.object_to_idx = {k: int(v) for k, v in self.object_to_idx.items()}
        if os.path.exists(relation_types_file):
            with open(relation_types_file, "r") as f:
                self.relation_to_idx = json.load(f)
                self.relation_to_idx = {k: int(v) for k, v in self.relation_to_idx.items()}

    def save_model(self, path):
        """
        Saves the model state dict with config in the filename.
        feature_encoder_{num_actions}_{rgb_dim}_{action_dim}_{map_dim}_{sg_dim}.pth
        Example: feature_encoder_45_512_32_64_256.pth
        """
        os.makedirs(path, exist_ok=True)
        # Gather relevant config parameters for filename
        num_actions = self.num_actions
        rgb_dim = self.rgb_encoder.output_dim
        action_dim = self.action_emb.embedding.embedding_dim
        sg_dim = self.lssg_encoder.lstm.hidden_size if not self.use_transformer else self.lssg_encoder.output_dim

        # Create filename including config parameters
        filename = f"feature_encoder_{num_actions}_{rgb_dim}_{action_dim}_{sg_dim}_{self.use_transformer}.pth"
        full_path = os.path.join(path, filename)
        # Save model state dict
        torch.save(self.state_dict(), full_path)

    @classmethod
    def create_and_load_model(cls, model_path, mapping_path=None, device="cpu"):
        """
        Loads a FeatureEncoder using parameters parsed from the model filename.
        Example filename: feature_encoder_45_512_32_64_256_False.pth
        feature_encoder_{num_actions}_{rgb_dim}_{action_dim}_{map_dim}_{sg_dim}.pth
        """
        # Parse parameters from the filename
        basename = os.path.basename(model_path)
        pattern = r"feature_encoder_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)\.pth"
        match = re.match(pattern, basename)
        if not match:
            raise ValueError(f"Filename {basename} does not match expected pattern.")

        num_actions = int(match.group(1))
        rgb_dim = int(match.group(2))
        action_dim = int(match.group(3))
        map_dim = int(match.group(4))
        sg_dim = int(match.group(5))

        # Instantiate the model with parsed parameters
        model = cls(
            num_actions=num_actions, rgb_dim=rgb_dim, action_dim=action_dim, map_dim=map_dim, sg_dim=sg_dim, mapping_path=mapping_path
        )

        # Load weights
        model.load_weights(model_path, device)
        return model

    def load_weights(self, model_path, device="cpu"):
        """
        Loads model weights into an existing FeatureEncoder instance.
        """
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        self.load_state_dict(state_dict)
        self.to(device)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = (
            nn.Conv2d(in_channels, out_channels, 1, stride=stride) if in_channels != out_channels or stride != 1 else nn.Identity()
        )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class OccupancyEncoder(nn.Module):
    def __init__(self, output_dim=64, num_orientations=4, include_agent_channel=True):
        super().__init__()
        self.include_agent_channel = include_agent_channel
        occ_channels = num_orientations
        in_channels = occ_channels + (1 if include_agent_channel else 0)

        self.encoder = nn.Sequential(ResidualBlock(in_channels, 32), nn.MaxPool2d(2), ResidualBlock(32, 64), nn.AdaptiveAvgPool2d((4, 4)))

        self.fc = nn.Linear(64 * 4 * 4, output_dim)

    def forward(self, occ_map, agent_pos=None):
        B, H, W, R = occ_map.shape
        C = R
        occ_map = occ_map.permute(0, 3, 1, 2).reshape(B, C, H, W)

        # Binary mask for valid (visited) areas
        valid_mask = (occ_map != -1).float()

        # Replace -1 with 0 for CNN input
        x = occ_map.clone()
        x[valid_mask == 0] = 0.0

        # Add agent position channel if enabled
        if self.include_agent_channel:
            agent_map = torch.zeros((B, 1, H, W), device=x.device)
            if agent_pos is not None:
                for b in range(B):
                    if agent_pos[b] is not None:
                        i, j = agent_pos[b]
                        if 0 <= i < H and 0 <= j < W:
                            agent_map[b, 0, i, j] = 1.0
            x = torch.cat([x, agent_map], dim=1)

        # Encode with CNN + residual blocks
        x = self.encoder(x)

        # Normalize features by valid coverage (avoid bias from padding)
        mask_resized = F.adaptive_avg_pool2d(valid_mask.sum(dim=1, keepdim=True), output_size=(4, 4))  # [B,1,4,4]
        x = x / mask_resized.clamp(min=1.0)
        x = x * (mask_resized > 0)  # Set output to 0, where there are no valid pixels

        return self.fc(x.view(B, -1))  # Final embedding vector


class ResNetFeatureExtractor(nn.Module):
    """
    Extracts visual features from the RGB input using pretrained ResNet18.
    Outputs a flat feature vector of size [B, 512].
    """

    def __init__(self, output_dim=512):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]  # remove FC layer
        self.resnet = nn.Sequential(*modules)
        self.projection = nn.Linear(512, output_dim) if output_dim != 512 else nn.Identity()
        self.output_dim = output_dim

    def forward(self, x):  # x: [B, 3, H, W]
        x = self.resnet(x)
        x = x.view(x.size(0), -1)  # [B, 512]
        return self.projection(x)  # [B, output_dim]


class ActionEmbedding(nn.Module):
    """
    Embeds the last discrete action into a learnable vector space.
    Index -1 is reserved for the 'START' state before any action is taken.
    """

    def __init__(self, num_actions, emb_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_actions + 1, emb_dim)  # 1 additional action for START (index: -1)

    def forward(self, action_idx):
        # Replace padding (-100) with 0 (arbitrary, due to masking later)
        safe_idx = action_idx.clone()
        safe_idx[safe_idx == -100] = 0
        return self.embedding(safe_idx + 1)  # [B, 32], plus 1 because of START action


class SceneGraphLSTMEncoder(nn.Module):
    """
    Encodes a sequence of scene graph embeddings using a two-layer LSTM.
    Supports both full sequences [B, T, D] and single steps [B, D].
    Optionally accepts/retruns LSTM hidden state for streaming/online inference.
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = 2
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=self.num_layers)

    def forward(self, x_seq, hidden=None):
        # Accepts [B, T, D] or [B, D]
        if x_seq.dim() == 2:
            # Single step: add time dimension
            x_seq = x_seq.unsqueeze(1)  # [B, 1, D]
        elif x_seq.dim() != 3:
            raise ValueError(f"Expected input [B, T, D] or [B, D], got {x_seq.shape}")

        # Pass through LSTM (optionally with hidden state)
        output, (hn, cn) = self.lstm(x_seq, hidden)  # output: [B, T, H], hn: [num_layers, B, H]

        return output, (hn, cn)


class SceneGraphTransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, num_heads=4, dropout=0.1, max_len=256):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_dim = input_dim

    def forward(self, x_seq, pad_mask=None):
        x_seq = self.input_proj(x_seq)
        x_seq = self.pos_encoder(x_seq)
        if pad_mask is not None:
            return self.transformer(x_seq, src_key_padding_mask=pad_mask)
        else:
            return self.transformer(x_seq)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [B, T, D]
        """
        return x + self.pe[:, : x.size(1)]
