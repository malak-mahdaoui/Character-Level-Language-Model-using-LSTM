# LSTM Text Generation Engine 
**PyTorch | Natural Language Processing | Recurrent Neural Networks**

## Project Overview
This project implements a **Character-Level Language Model** designed to learn the linguistic style and structure of a provided text corpus (e.g., Voltaire's *Candide*). By treating text generation as a sequence prediction problem, the model learns to predict the next character in a sequence, allowing it to generate entirely new, human-like text.

##  Key Features
* **Architecture:** Multi-layer **LSTM** network with an integrated **Embedding** layer and a Linear output head.
* **Text Processing:** Custom `CharDataset` implementing a sliding window approach to create (input, target) pairs for training.
* **Controlled Generation:** Advanced sampling function with **Temperature control** (0.1 - 1.5) and **Top-K filtering** to balance creativity and coherence.
* **Training Pipeline:** Robust loop featuring `CrossEntropyLoss`, `Adam` optimization, and **Gradient Clipping** to prevent instability.
* **Persistence:** Full model serialization (checkpoints) and JSON-based vocabulary mapping for easy deployment.

##  Technical Stack
* **Framework:** PyTorch (v2.x)
* **Environment:** Optimized for CUDA/GPU acceleration with CPU fallback.
* **Data Handling:** NumPy & Matplotlib for loss visualization and dataset management.
