import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_scatter import scatter_mean
from layers import SparseHypergraphLayer, CausalDeconfoundingLayer
from transformers import AutoTokenizer, AutoModel  # For BioBERT
# from rdkit import Chem  # For SMILES to graph (uncomment for structural)

class GINEncoder(nn.Module):
    """Encodes 3D molecular structures (SMILES graphs). Improved: Add edge features if available."""
    def __init__(self, num_features, hidden_dim):
        super(GINEncoder, self).__init__()
        nn1 = nn.Sequential(nn.Linear(num_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        nn2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        return global_mean_pool(x, batch)

class CrossModalAttention(nn.Module):
    """Learns weights to fuse Structural (GNN) and Semantic (BioBERT) embeddings. Added multi-head."""
    def __init__(self, dim, num_heads=4):
        super(CrossModalAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.query = nn.Linear(dim, num_heads * self.head_dim)

    def forward(self, z_str, z_sem):
        stacked = torch.stack([z_str, z_sem], dim=1)  # [B, 2, D]
        scores = self.query(stacked).view(-1, 2, self.num_heads, self.head_dim).mean(-1)  # Multi-head avg [B, 2, H]
        weights = F.softmax(scores, dim=1)  # [B, 2, H]
        fused = torch.sum(weights.unsqueeze(-1) * stacked.unsqueeze(2), dim=1)  # Weighted sum
        return fused.squeeze(1)  # [B, D]

class HyperCausalDDI(nn.Module):
    def __init__(self, n_drugs, str_input_dim, sem_input_dim, hidden_dim, n_side_effects, n_prototypes=20):
        super(HyperCausalDDI, self).__init__()
        
        # 1. Encoders - Improved: Load BioBERT once
        self.semantic_model = AutoModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
        self.semantic_model.eval()  # Frozen
        self.semantic_proj = nn.Linear(sem_input_dim, hidden_dim)
        self.structural_encoder = GINEncoder(str_input_dim, hidden_dim)
        
        # Fusion - Multi-head attention
        self.fusion = CrossModalAttention(hidden_dim)
        
        # 2. Hypergraph Learning - Stacked layers
        self.hyper_conv1 = SparseHypergraphLayer(hidden_dim, hidden_dim)
        self.hyper_conv2 = SparseHypergraphLayer(hidden_dim, hidden_dim)
        
        # 3. Causal Inference Heads (Multi-task for each side effect)
        self.causal_heads = nn.ModuleList([
            CausalDeconfoundingLayer(hidden_dim, n_prototypes) 
            for _ in range(n_side_effects)
        ])
        
        # Embedding lookup for drugs (nodes in hypergraph)
        self.drug_embedding = nn.Embedding(n_drugs, hidden_dim)

    def forward(self, structure_data, semantic_emb, hyperedge_index, batch_indices,