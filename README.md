# HyperCausal-DDI: De-confounded High-Order Polypharmacy Side Effect Prediction

This repository implements the HyperCausal-DDI model from the paper ["Beyond Associations: HyperCausal-DDI for De-confounded High-Order Polypharmacy Side Effect Prediction"](link-to-paper-if-published) (Smart Health, 2025).

## **📌 Overview**
HyperCausal-DDI integrates hypergraph neural networks with structural causal models to predict high-order drug interactions while de-confounding patient comorbidities. 

### **Key Features:**

* **Hypergraph Formalism:** Models drug combinations as hyperedges to capture non-additive synergistic effects.  
* **Causal De-confounding:** Implements a *Neural Backdoor Adjustment* layer to remove confounding bias (e.g., patient comorbidities) from observational EHR data.  
* **Multi-Modal Fusion:** Integrates 3D molecular structures (GNN) with text-based Mechanism of Action (BioBERT).  
* **Scalable:** Utilizes sparse matrix operations to scale linearly with the number of prescriptions.


## **📂 Repository Structure**

.  
├── layers.py           \# Custom layers: CausalDeconfounding & SparseHypergraphConv  
├── model.py            \# Main architecture: HyperCausalDDI & Multi-modal Encoders  
├── train.py            \# Training loop, Focal Loss, and Data Loading logic  
├── requirements.txt    \# Python dependencies  
└── README.md           \# Project documentation

## **🛠 Installation**

High-order graph learning requires specific versions of PyTorch Geometric. Please follow these steps carefully to avoid compilation errors.

### **1\. Create Environment**

```bash
conda create \-n hyperddi python=3.9  
conda activate hyperddi
```

### **2\. Install PyTorch (GPU version recommended)**

Adjust the CUDA version (cu118) based on your system:

```bash
pip install torch torchvision torchaudio \--index-url \[https://download.pytorch.org/whl/cu118\](https://download.pytorch.org/whl/cu118)
```

### **3\. Install Graph Dependencies**

**Critical Step:** These libraries must match your PyTorch and CUDA versions.

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \-f \[https://data.pyg.org/whl/torch-2.0.0+cu118.html\](https://data.pyg.org/whl/torch-2.0.0+cu118.html)  
pip install torch-geometric
```

### **4\. Install Other Requirements**

```bash
git clone https://github.com/tanhaei/HyperCausal-DDI.git
cd HyperCausal-DDI
pip install \-r requirements.txt
```

## **📊 Data Preparation**

Due to data privacy regulations (HIPAA), we cannot release the raw patient data. You must acquire credentials for MIMIC-IV.

### **1\. Clinical Data (MIMIC-IV)**

1. Sign the Data Use Agreement at [PhysioNet](https://physionet.org/content/mimiciv/).  
2. Download the prescriptions and diagnoses\_icd tables.  
3. Pre-process them to generate:  
   * hyperedge\_index.pt: A sparse tensor \[2, N\_connections\] mapping drugs to prescriptions.  
   * labels.pt: Binary vectors \[N\_prescriptions, 65\] for side effects.

### **2\. Molecular Data (DrugBank)**

1. Download structure data from [DrugBank](https://go.drugbank.com/).  
2. Convert SMILES to PyG graphs using RDKit.  
3. Extract MoA text and embed using BioBERT.

*Note: The current train.py script includes a dummy data generator for debugging purposes. Uncomment the data loading section to use real data.*

## **🚀 Usage**

To train the model on the default configuration:

```bash
python train.py
```

### **Hyperparameters**

You can modify the following parameters in train.py:

* N\_DRUGS: Number of unique drugs in your dataset.  
* N\_SIDE\_EFFECTS: Number of adverse events to predict (default: 65).  
* K\_PROTOTYPES: Number of confounder clusters for causal adjustment (default: 20).

## **⚖️ Citation**

If you use this code or our methodology in your research, please cite:

## **🤝 Acknowledgments**

* **MIMIC-IV Team** for providing the clinical datasets.  
* **PyTorch Geometric** for the graph learning primitives.

## **📝 License**

This project is licensed under the MIT License \-
