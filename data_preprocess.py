"""
data_preprocess.py: Preprocess MIMIC-IV + DrugBank for HyperCausal-DDI.
Based on paper: Curate hyperedges from prescriptions, map to DrugBank, cluster confounders.
Usage: python data_preprocess.py --mimic_dir ./data/mimic/ --drugbank_key YOUR_KEY --output_dir ./processed/
"""

import argparse
import pandas as pd
import requests
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp
import os
from datetime import datetime

# Top-65 adverse events (example from paper: AKI, Hypotension, etc. - map to ICD codes)
TOP_ADVERSE_EVENTS = [
    'N17.0',  # AKI
    'R57.0',  # Hypotension
    # ... add 63 more ICD-10 codes for top adverse events
] * 65  # Placeholder - replace with actual top-65 from MIMIC

TOP_ICD_CODES = 50  # Top-50 ICD for confounders
K_CONF = 20  # K for K-Means++

def download_drugbank(drugbank_key, output_dir):
    """Download DrugBank data (SMILES, MoA) via API."""
    url = "https://api.drugbank.com/v1/drugs"
    headers = {"Authorization": f"Bearer {drugbank_key}"}
    params = {"include": "smiles,description"}  # MoA in description
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        drugs = response.json()["drugs"]
        df_drugs = pd.DataFrame(drugs)
        df_drugs['drugbank_id'] = df_drugs['drugbank_id']
        df_drugs['smiles'] = df_drugs['smiles']
        df_drugs['moa'] = df_drugs['description']  # Extract MoA from description
        df_drugs.to_csv(os.path.join(output_dir, 'drugbank.csv'), index=False)
        print(f"Downloaded {len(df_drugs)} drugs from DrugBank.")
        return df_drugs
    else:
        raise ValueError(f"DrugBank API error: {response.status_code}")

def load_mimic_prescriptions(mimic_dir):
    """Load prescriptions from MIMIC-IV CSV."""
    df_pres = pd.read_csv(os.path.join(mimic_dir, 'prescriptions.csv'))
    df_pres['admittime'] = pd.to_datetime(df_pres['admittime'])  # For temporal split
    df_pres['drug_name'] = df_pres['drug_name'].str.lower()  # Normalize
    return df_pres

def load_mimic_diagnoses(mimic_dir):
    """Load ICD diagnoses for labels and confounders."""
    df_diag = pd.read_csv(os.path.join(mimic_dir, 'diagnoses_icd.csv'))
    df_diag['icd_code'] = df_diag['icd_code'].str[:3]  # ICD-10 level
    return df_diag

def map_drugs_to_drugbank(df_pres, df_drugs):
    """Map RxNorm/drug names to DrugBank IDs (simplified - use UMLS in production)."""
    # Placeholder mapping - in real, use RxNorm API or UMLS
    mapping = {}  # e.g., {'lisinopril': 'DB00722'}
    df_pres['drugbank_id'] = df_pres['drug_name'].map(mapping)
    df_pres = df_pres.dropna(subset=['drugbank_id'])
    return df_pres.merge(df_drugs, on='drugbank_id')

def create_hyperedges(df_pres, min_drugs=3):
    """Create hyperedges from prescriptions (combinations with >=3 drugs)."""
    hyperedges = []
    for _, group in df_pres.groupby('subject_id'):  # Per patient admission
        drugs = group['drugbank_id'].unique()
        if len(drugs) >= min_drugs:
            # Create subsets of size 3+ as hyperedges (or full set)
            from itertools import combinations
            for combo in combinations(drugs, min_drugs):  # e.g., triads
                hyperedges.append(list(combo))
    return hyperedges

def extract_labels(df_diag, subject_ids):
    """Extract multi-label side effects (top-65 adverse from ICD)."""
    df_diag = df_diag[df_diag['subject_id'].isin(subject_ids)]
    mlb = MultiLabelBinarizer(classes=TOP_ADVERSE_EVENTS)
    labels = mlb.fit_transform(df_diag['icd_code'].tolist())
    return torch.tensor(labels, dtype=torch.float)  # Shape: (num_samples, 65)

def cluster_confounders(df_patients, k=K_CONF):
    """Cluster confounders: demographics + top-50 ICD."""
    # Assume df_patients has 'age', 'gender' (one-hot), 'icd_codes' (multi-hot top-50)
    scaler = MinMaxScaler()
    features = scaler.fit_transform(df_patients[['age', 'gender']])  # Normalize
    # Add ICD multi-hot (placeholder)
    icd_features = pd.get_dummies(df_patients['icd_code']).iloc[:, :TOP_ICD_CODES].fillna(0).values
    X = np.hstack([features, icd_features])
    
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    prototypes = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    print(f"Confounder prototypes (K={k}): {centroids.shape}")
    return centroids, prototypes  # Centroids for C, labels for assignment

def temporal_split(df_pres):
    """Split by time: 2008-2016 train, 2017 val, 2018-2019 test."""
    df_pres['year'] = df_pres['admittime'].dt.year
    train = df_pres[df_pres['year'] <= 2016]
    val = df_pres[df_pres['year'] == 2017]
    test = df_pres[df_pres['year'] >= 2018]
    return train, val, test

def save_hypergraph(hyperedges, node_features, edge_index, y, split, output_dir):
    """Save as PyG Data (incidence matrix for hypergraph)."""
    # Incidence H: rows=drugs (|V|), cols=hyperedges (|E|)
    V = len(node_features)  # Assume node_features shape (N_drugs, d)
    E = len(hyperedges)
    H_data = []
    H_row = []
    H_col = []
    for e_idx, edge in enumerate(hyperedges):
        for v_idx, v in enumerate(edge):
            H_row.append(v)  # Drug ID as index
            H_col.append(e_idx)
            H_data.append(1)
    
    H = sp.csr_matrix((H_data, (H_row, H_col)), shape=(V, E))
    edge_index, _ = from_scipy_sparse_matrix(H)  # PyG format
    
    data = Data(x=torch.tensor(node_features), edge_index=edge_index, y=y)
    data.split = split  # {'train': idx, etc.}
    torch.save(data, os.path.join(output_dir, f'hypergraph_{split}.pt'))

def main(args):
    # Download DrugBank
    df_drugs = download_drugbank(args.drugbank_key, args.output_dir)
    
    # Load MIMIC
    df_pres = load_mimic_prescriptions(args.mimic_dir)
    df_diag = load_mimic_diagnoses(args.mimic_dir)
    
    # Map to DrugBank
    df_pres = map_drugs_to_drugbank(df_pres, df_drugs)
    
    # Temporal split
    train_pres, val_pres, test_pres = temporal_split(df_pres)
    
    # Create hyperedges (example for train)
    train_hyper = create_hyperedges(train_pres)
    # Similar for val/test
    
    # Extract labels (per hyperedge - map to patient admission)
    subject_ids = df_pres['subject_id'].unique()
    y = extract_labels(df_diag, subject_ids)
    
    # Confounder clustering (on patient df - assume df_patients from admissions.csv)
    df_patients = pd.read_csv(os.path.join(args.mimic_dir, 'admissions.csv'))  # For age/gender
    # Merge with ICD for features
    centroids, _ = cluster_confounders(df_patients)
    np.save(os.path.join(args.output_dir, 'confounders.npy'), centroids)
    
    # Node features: Concat structural (placeholder) + semantic (MoA embeddings - use BioBERT in full)
    node_features = np.random.rand(len(df_drugs), 128)  # Placeholder - replace with GIN/BioBERT
    # Edge index from hyperedges
    
    # Save
    save_hypergraph(train_hyper, node_features, None, y[:len(train_hyper)], 'train', args.output_dir)
    print("Preprocessing complete! Files saved to", args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mimic_dir', type=str, default='./data/mimic/')
    parser.add_argument('--drugbank_key', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./processed/')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
