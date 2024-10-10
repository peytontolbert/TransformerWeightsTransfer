# Weights Transfer Playground

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Overview

The **Weights Transfer Playground** is an experimental platform designed for cross-architecture analysis and weight transfer between diverse neural network models. This project facilitates the exploration of transferring weights from one architecture to another, enabling comparative studies and enhancing model performance through knowledge sharing.

## Features

- **Cross-Architecture Weight Transfer**: Transfer weights between different neural network architectures such as Transformers, LSTMs, Mamba, and LiquidS4.
- **Unified Evaluation Framework**: Evaluate multiple model architectures on synthetic and real-world datasets with comprehensive metrics.
- **Visualization Tools**: Visualize the convergence and differences in weights across various checkpoints and architectures using PCA and graph-based methods.
- **Extensible Architecture**: Easily integrate additional models and extend existing transfer mechanisms to accommodate new architectures.
- **Automated Scripts**: Utilize scripts for weight extraction, conversion, evaluation, and visualization to streamline experimentation.

## Installation

To set up the Weights Transfer Playground, ensure you have **Python 3.8+** installed. The project relies on several Python packages, including PyTorch and Hugging Face's Transformers library.

### Clone the Repository

```
git clone https://github.com/your-username/weights-transfer-playground.git
cd weights-transfer-playground
```

### Install Dependencies

You can install the required packages using `pip`:
```bash
pip install torch transformers matplotlib networkx scikit-learn fire numpy
```

Alternatively, if you prefer using `requirements.txt`, create the file with the necessary dependencies and install them:

```bash
pip install -r requirements.txt
```


*Note: Ensure you have access to a GPU for efficient model training and inference.*

## Usage

### Running the Main Script

The `main.py` script demonstrates the transfer of weights from a pretrained LLaMA model to various target architectures. It processes a sample input text and compares outputs between the transferred models and the original LLaMA model.


```bash
python main.py
```


### Model Conversion

To convert a LLaMA model to a different architecture (e.g., LiquidS4), use the corresponding script in the `convert/` directory:
```bash
python convert/convert_llama_to_liquid_s4.py
```


### Evaluation

Evaluate different model architectures on synthetic datasets using the unified evaluation script:

```bash
python scripts/unifiedeval.py
```


### Viewing Checkpoints and Graphs

Visualize the convergence of model weights across checkpoints or view the morphism graph:
```bash
python eval/view_checkpoints.py
python eval/view_graph.py
```


## Project Structure
```bash
weights-transfer-playground/
├── README.md
├── main.py
├── llamaconfig
├── scripts/
│ ├── compute_distances.py
│ ├── dataset.py
│ ├── define_morphisms.py
│ ├── eval.py
│ ├── extract_weights.py
│ ├── unifiedeval.py
│ ├── unifiedmodel.py
│ └── view_graph.py
├── convert/
│ │ ├── convert_llama_to_liquid_s4.py
│ │ ├── convert_refined_conversion.py
│ │ ├── convert_llamatomamba.py
│ │ └── convert_llamatotransformer.py
├── eval/
│ ├── unifiedeval.py
│ ├── view_checkpoints.py
│ └── view_graph.py
├── checkpoints/
│ ├── checkpoint_epoch_100.pth
│ ├── checkpoint_epoch_500.pth
│ └── checkpoint_epoch_1000.pth
├── extracted_weights.npy
├── model_graph.gpickle
└── .gitignore
```

### Detailed File Descriptions

#### Root Directory
- **`README.md`**: Provides an overview and instructions for the project.
- **`main.py`**: The main script that demonstrates weight transfer between models.

#### `llamaconfig/`
- Contains configuration files for the LLaMA model.

#### `scripts/`
- **`convert/`**
  - **`convert_llama_to_liquid_s4.py`**: Converts a LLaMA model to a LiquidS4 model.
  - **`convert_refined_conversion.py`**: Refines the weight transfer process between models.
  - **`convert_llamatomamba.py`**: Converts LLaMA model to Mamba architecture.
  - **`convert_llamatotransformer.py`**: Converts LLaMA model to a Transformer architecture.
  
- **`compute_distances.py`**: Computes Fisher Information for each checkpoint.
- **`dataset.py`**: Defines synthetic sequence datasets for training and evaluation.
- **`define_morphisms.py`**: Defines morphisms for the model graph.
- **`eval.py`**: Evaluates different models on test datasets.
- **`extract_weights.py`**: Extracts weights from model checkpoints.
- **`unifiedeval.py`**: Unified evaluation script.
- **`unifiedmodel.py`**: Defines the `UnifiedModel` integrating various architectures.
- **`view_graph.py`**: Script for viewing the morphism graph.

#### `convert/`
- Contains additional model conversion scripts as needed.

#### `eval/`
- **`unifiedeval.py`**: Evaluates multiple model architectures on synthetic datasets.
- **`view_checkpoints.py`**: Extracts and visualizes weights from model checkpoints.
- **`view_graph.py`**: Visualizes the morphism graph between checkpoints.

#### `visualization/`
- **`visualize_weight_space.py`**: Visualizes the convergence of architecture weights using PCA.

#### `checkpoints/`
- Stores model checkpoint files for different training epochs.

#### Other Files
- **`extracted_weights.npy`**: Numpy file containing extracted weights from checkpoints.
- **`model_graph.gpickle`**: Pickle file storing the morphism graph.
- **`.gitignore`**: Specifies files and directories to ignore in Git.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. **Fork the Repository**: Click the fork button at the top right of this page.
2. **Clone Your Fork**:
    ```bash
    git clone https://github.com/peytontolbert/weights-transfer-playground.git
    cd weights-transfer-playground
    ```
3. **Create a Branch**:
    ```bash
    git checkout -b feature/YourFeatureName
    ```
4. **Make Your Changes**.
5. **Commit Your Changes**:
    ```bash
    git commit -m "Add some feature"
    ```
6. **Push to Your Fork**:
    ```bash
    git push origin feature/YourFeatureName
    ```
7. **Open a Pull Request**: Navigate to the original repository and open a pull request.

Please ensure your contributions adhere to the project’s coding standards and include appropriate tests.

## Acknowledgements

- [LLaMA](https://github.com/facebookresearch/llama) by Facebook Research for the foundational model.
- [PyTorch](https://pytorch.org/) for the deep learning framework.
- [Hugging Face Transformers](https://github.com/huggingface/transformers) for model and tokenizer utilities.
- [NetworkX](https://networkx.org/) and [Matplotlib](https://matplotlib.org/) for graphing and visualization tools.
- [Scikit-learn](https://scikit-learn.org/) for evaluation metrics and PCA.
- [Fire](https://github.com/google/python-fire) for creating CLIs.

---

*Feel free to reach out with any questions or suggestions!*