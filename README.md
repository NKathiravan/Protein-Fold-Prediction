# 🧬 Protein Fold and Secondary Structure Prediction using ProtBERT, Machine Learning, and Quantum Hybrid Models

This project explores **Protein Fold Prediction** and **Protein Secondary Structure Prediction (PSSP)** using advanced techniques such as **transformer-based embeddings (ProtBERT)**, classical **supervised machine learning models**, and a **Quantum-Classical Hybrid MLP architecture**. 

It aims to classify proteins into distinct fold types and predict structural elements like **alpha-helices** and **beta-sheets**, offering insights into their biological functions.

---

## 📌 Key Features

- ⚛️ **ProtBERT-based Embedding** for protein sequence representation.
- 🧠 **Machine Learning Models** for classification of protein folds and secondary structure.
- 🔬 **Quantum Hybrid MLP** to integrate quantum computing for enhanced prediction.
- 📊 Comprehensive evaluation using **accuracy**, **precision**, **recall**, and **F1-score**.
- 🧼 Effective preprocessing of **CASP datasets** for PSSP.
- 📈 Feature selection and dimensionality reduction for improved performance.

---

## 🧱 Architecture Overview

### 1️⃣ Protein Fold Prediction

- **Input**: Raw protein sequences.
- **Embedding**: `ProtBERT` transforms sequences into 1024-dimensional embeddings.
- **Models Used**:
  - Single-layer MLP (Input → 100 neurons → Output)
  - Multi-layer MLP (Input → 128 → 4 → Output)
  - Hybrid MLP (Input → 128 classical → 4 quantum qubits → Output)
- **Quantum Component**: The Hybrid MLP integrates a quantum circuit using 4 qubits to capture complex patterns beyond classical capabilities.

### 2️⃣ Protein Secondary Structure Prediction (PSSP)

- **Input**: CASP dataset (after cleaning and formatting).
- **Feature Engineering**: Automated using ProtBERT + manual selection for key features.
- **Prediction**: Classification into secondary structure types (e.g., α-helix, β-sheet).
- **Evaluation**: Performance assessed using standard classification metrics.

---

## 🔄 Data Preprocessing

- **ProtBERT Embedding**:
  - Transformer-based model trained on massive protein sequence data.
  - Learns amino acid relationships and encodes sequences into feature-rich vectors.
  - Automates feature engineering, replacing manual approaches.
  - Enhances training stability through normalization.

- **Preprocessing for PSSP**:
  - Data cleaning and formatting.
  - Feature selection to retain key structural signals while reducing dimensionality.

---

## 🏗️ Model Architecture

### MLP Variants

| Model Type         | Layers/Neurons                   | Notes                                      |
|--------------------|----------------------------------|--------------------------------------------|
| Single-layer MLP   | Input: 1024 → Hidden: 100 → Out  | Basic model for pattern extraction         |
| Multi-layer MLP    | Input → 128 → 4 → Output         | Captures complex non-linear relationships |
| Hybrid MLP         | Input → 128 (classical) + 4 Qubits → Output | Quantum-enhanced feature learning |

---

## 📊 Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

Each model is evaluated using these metrics to determine performance on both fold classification and secondary structure prediction tasks.

---

## 🚀 Getting Started

### 🔧 Requirements

- Python 3.8+
- `transformers`, `scikit-learn`, `qiskit`, `pennylane`, `torch`, `numpy`, `matplotlib`

### 🛠 Installation

```bash
pip install -r requirements.txt
