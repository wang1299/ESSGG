from transformers import BertTokenizer, BertModel
import os

save_directory = "/home/wgy/bert-base-uncased"

print(f"Downloading bert-base-uncased to {save_directory}...")
try:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)
    print("Download success!")
except Exception as e:
    print(f"Download failed: {e}")
