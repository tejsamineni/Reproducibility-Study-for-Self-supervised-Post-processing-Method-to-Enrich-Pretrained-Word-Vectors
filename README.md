# Reproducibility Study for Self-supervised Post-processing Method to Enrich Pretrained Word Vectors

This repository contains the implementation and analysis for the reproducibility study of the paper **"Self-supervised Post-processing Method to Enrich Pretrained Word Vectors."** The paper proposes a method called **Self-Extrofitting**, which enriches pretrained static word embeddings without relying on external lexicons. This project aims to reproduce the results from the original paper, evaluate the claims, and provide insights into potential improvements.

---

## Description

The project focuses on:
- **Self-Extrofitting:** Enriching pretrained word embeddings using self-supervised techniques.
- **Comparison:** Validating the reproduced results against the original paper and analyzing differences.
- **Downstream Tasks:** Evaluating the enriched embeddings on text classification tasks.

We implemented and analyzed the methods described in the paper, including:
1. Semantic similarity extraction.
2. Extrofitting word embeddings.
3. Text classification tasks using enriched embeddings.

---

## Files and Folders

### Main Files:
- **`selfextro_mv-2.ipynb`:** Combines semantic similarity extraction and extrofitting processes. This notebook is the core implementation of Self-Extrofitting, including batch-wise SVD for scalability.
- **`Text_Classification_mv.ipynb`:** Implements text classification tasks using the enriched embeddings from the Self-Extrofitting process. The classification is performed on the AGNews dataset.
- **`utils-2.py`:** Utility functions for data preprocessing, embedding handling, and dataset preparation.
- **`TextCNN.py`:** Implements a TextCNN model for text classification tasks.
- **`dataloader.py`:** Contains helper functions to load datasets into PyTorch dataloaders.

### Dataset:
- **`AGNews.zip`:** The dataset used for the classification tasks, containing labeled news articles. Extract this file before running the text classification notebook.

---

## Running the Code

### Prerequisites
- Python 3.x
- PyTorch
- Required libraries: `numpy`, `sklearn`, `tqdm`, `torchvision`, `pandas`

### Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/tejsamineni/Reproducibility-Study-for-Self-supervised-Post-processing-Method-to-Enrich-Pretrained-Word-Vectors.git
   cd Reproducibility-Study-for-Self-supervised-Post-processing-Method-to-Enrich-Pretrained-Word-Vectors

   ### Perform Self-Extrofitting:
2. Open and run the `selfextro_mv-2.ipynb` notebook.
Ensure that:
   - File paths are updated to match your local directory structure.
   - Batch size is adjusted based on your system's GPU memory and computational capacity.
This notebook performs:
   - Semantic similarity extraction.
   - Extrofitting pretrained embeddings.
The enriched embeddings are saved to a file (e.g., `fastTextSelfExtro_threshold95_dim300.txt`).

### Evaluate Enriched Embeddings:
3. Open and run the `Text_Classification_mv.ipynb` notebook.
 Ensure that:
   - The file path for the enriched embeddings generated in the previous step is correctly updated.
   - The batch size is appropriate for your system's GPU capabilities.
This notebook performs:
   - Text classification on the AGNews dataset.
   - Validation and test accuracy reporting.
