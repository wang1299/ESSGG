# Learning to Explore: Reinforcement Learning with Scene Graph Integration for Full Environment Coverage

This repository contains the official source code for the Master's Thesis by Roman Kübler, submitted to the University of Augsburg. The project investigates the impact of modern reinforcement learning algorithms and network architectures on the task of Embodied Semantic Scene Graph Generation (ESSGG).

---

## 📖 Overview

This work addresses the challenge of semantic exploration for embodied agents. The primary goal is to build a complete and accurate semantic graph of an unknown environment. We systematically evaluate the replacement of a baseline navigation module (using REINFORCE and LSTM) with more advanced components:

* **Algorithmic Modernization:** Substituting the high-variance **REINFORCE** algorithm with a more stable **Advantage Actor-Critic (A2C)** agent.
* **Architectural Modernization:** Replacing a recurrent **LSTM**-based model with a **Transformer**-based architecture to better handle long-term dependencies.

The entire framework is implemented and evaluated in the **AI2-THOR** simulation environment.

---

## ✨ Key Features

* Implementation of **A2C** and **REINFORCE** agents for the ESSGG navigation task.
* Modular design allowing for both **LSTM** and **Transformer** backbones.
* A comprehensive **2x2 ablation study** to systematically evaluate each component.
* Scripts for generating expert datasets, pre-training, hyperparameter optimization, training, and evaluation.
* Integration with a pre-computed environment database for faster, decoupled training runs.

---

## 🔧 Installation

To set up the environment and install the required dependencies, follow these steps. It is recommended to use a virtual environment (e.g., conda or venv).

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kueblero/Embodied-Semantic-Scene-Graph-Generation.git
    cd Embodied-Semantic-Scene-Graph-Generation
    ```

2.  **Install dependencies:**
    The required Python packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This project was developed using Python 3.9+ and PyTorch.*

3.  **AI2-THOR Environment:**
    This project relies on a pre-computed database of the AI2-THOR environment scenes. The scripts to generate this database are included.
    
---

## 🚀 Reproducing the Results

The experimental results presented in the thesis can be reproduced in several ways, depending on your goal. This repository includes all necessary pre-computed data (transition tables, IL dataset, encoder weights, and final agent weights) to allow you to skip certain steps.

You can choose one of the following paths:
* **[Option A] Evaluate Final Models:** The quickest path to verify the final performance metrics using the provided trained agents.
* **[Option B] Re-run the RL Training:** Start from the pre-trained encoders to run the hyperparameter optimization and main RL training yourself.
* **[Option C] Full Reproduction from Scratch:** Execute the entire pipeline, including data generation and encoder pre-training.

### Option A: Evaluate Final Models (Quickest)

This option uses the final trained agent weights provided in this repository to run the evaluation on the test set.

```bash
# Example: Evaluate the provided A2C-LSTM agent
python RL_training/evaluate_model_weights.py --conf_path RL_training/sbatch/sl_configs/A2C_LSTM
```
Repeat this command for the other three configurations (`A2C_Transformer`, `REINFORCE_LSTM`, `REINFORCE_Transformer`) to obtain all final evaluation results from the thesis.

### Option B: Re-run the RL Training

This option is for you if you want to run the main reinforcement learning phase yourself. It uses the provided pre-computed environment, IL dataset, and pre-trained encoder weights.

1.  **Run Hyperparameter Optimization:**
    Execute the optimization scripts to find the optimal values for `ρ`, value, and entropy coefficients.
    ```bash
    # Optimize for A2C + LSTM
    python optim/a2c_param_optimizer.py
    
    # Optimize for A2C + Transformer
    python optim/a2c_param_optimizer.py --use_transformer
    
    # ... and so on for the REINFORCE variants.
    ```
    Then, you **must** manually enter the resulting values into the corresponding configuration files in `RL_training/sbatch/configs/`.

2.  **Run Main RL Training:**
    First, generate the final configs, then start the training runs.
    ```bash
    # 1. Create final configs
    python RL_training/sbatch/create_sl_configs.py

    # 2. Run training for one agent (e.g., A2C + LSTM)
    python RL_training/main.py --conf_path RL_training/sbatch/sl_configs/A2C_LSTM --precomputed --save_model
    ```
    Repeat the training run for all four agent configurations.

### Option C: Full Reproduction from Scratch

This option executes the entire pipeline from the very beginning.

1.  **Environment Pre-computation:**
    ```bash
    python components/scripts/generate_transition_tables.py
    ```

2.  **Imitation Learning & Encoder Pre-training:**
    Generate the IL dataset, pre-train the encoder (once for LSTM, once for Transformer by setting `"use_transformer": true/false` in `config/navigation.json`), and update the encoder path in `config/agent.json`.
    ```bash
    python ImitationLearning/scripts/generate_il_dataset.py
    python ImitationLearning/train_il.py
    ```

3.  **Hyperparameter Optimization, RL Training, and Evaluation:**
    Follow the steps outlined in **Option B** and then **Option A**'s evaluation step to complete the process.

### 📈 Viewing Results

If you run the RL training (Options B or C), the progress is logged and can be visualized using TensorBoard.
```bash
tensorboard --logdir RL_training/runs
```
Navigate to `localhost:6006` in your browser to view the plots.

---

## 📜 Citation

If you find this work useful in your research, please consider citing the thesis:

```bibtex
@mastersthesis{kueble2025learning,
  author       = {Roman Küble},
  title        = {Learning to Explore: Reinforcement Learning with Scene Graph Integration for Full Environment Coverage},
  school       = {University of Augsburg},
  year         = {2025},
  month        = {September}
}
```

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
