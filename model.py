import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from layers import SparseHypergraphLayer, CausalDeconfoundingLayer

class GINEncoder(nn.Module):
    """Encodes 3D molecular structures (SMILES graphs)."""
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
    """Learns weights to fuse Structural (GNN) and Semantic (BioBERT) embeddings."""
    def __init__(self, dim):
        super(CrossModalAttention, self).__init__()
        self.query = nn.Linear(dim, 1)
        
    def forward(self, z_str, z_sem):
        # Stack: [Batch, 2, Dim]
        stacked = torch.stack([z_str, z_sem], dim=1) 
        # Calculate attention scores
        scores = self.query(stacked) # [Batch, 2, 1]
        weights = F.softmax(scores, dim=1) # [Batch, 2, 1]
        
        # Weighted sum
        fused = (weights[:, 0] * z_str) + (weights[:, 1] * z_sem)
        return fused

class HyperCausalDDI(nn.Module):
    def __init__(self, n_drugs, str_input_dim, sem_input_dim, hidden_dim, n_side_effects, n_prototypes=20):
        super(HyperCausalDDI, self).__init__()
        
        # 1. Encoders
        # Note: We assume BioBERT embeddings are pre-computed (frozen) for efficiency
        self.semantic_proj = nn.Linear(sem_input_dim, hidden_dim)
        self.structural_encoder = GINEncoder(str_input_dim, hidden_dim)
        
        # Fusion
        self.fusion = CrossModalAttention(hidden_dim)
        
        # 2. Hypergraph Learning
        self.hyper_conv1 = SparseHypergraphLayer(hidden_dim, hidden_dim)
        self.hyper_conv2 = SparseHypergraphLayer(hidden_dim, hidden_dim)
        
        # 3. Causal Inference Heads (Multi-task for each side effect)
        # We create a list of Causal Layers, one for each top-K side effect
        self.causal_heads = nn.ModuleList([
            CausalDeconfoundingLayer(hidden_dim, n_prototypes) 
            for _ in range(n_side_effects)
        ])
        
        # Embedding lookup for drugs (nodes in hypergraph)
        # In a real scenario, this is initialized with fused features
        self.drug_embedding = nn.Embedding(n_drugs, hidden_dim)

    def forward(self, structure_data, semantic_emb, hyperedge_index, batch_indices):
        """
        structure_data: Batch of molecular graphs
        semantic_emb: Pre-computed BioBERT embeddings [N_drugs, 768]
        hyperedge_index: [2, N_edges] sparse incidence matrix (node_idx, hyperedge_idx)
        batch_indices: indices of hyperedges in the current batch
        """
        
        # --- Step 1: Feature Encoding ---
        # z_str = self.structural_encoder(structure_data.x, structure_data.edge_index, structure_data.batch)
        # z_sem = self.semantic_proj(semantic_emb)
        # node_features = self.fusion(z_str, z_sem)
        
        # *Simplification for code skeleton*: using learned embeddings directly
        node_features = self.drug_embedding.weight 
        
        # --- Step 2: Hypergraph Convolution ---
        # Propagate info to update node embeddings based on high-order connections
        x = self.hyper_conv1(node_features, hyperedge_index)
        x = self.hyper_conv2(x, hyperedge_index)
        
        # --- Step 3: Pooling to Hyperedge Level (Treatment T) ---
        # We need to pool node embeddings into hyperedge embeddings
        # hyperedge_index[0] = nodes, hyperedge_index[1] = hyperedges
        # We use scatter_mean to aggregate nodes belonging to the same hyperedge
        from torch_scatter import scatter_mean
        treatment_emb = scatter_mean(x[hyperedge_index[0]], hyperedge_index[1], dim=0)
        
        # Select only the hyperedges in the current batch
        batch_treatment = treatment_emb[batch_indices]
        
        # --- Step 4: Causal De-confounding & Prediction ---
        outputs = []
        for head in self.causal_heads:
            # Each head predicts probability for one side effect (e.g., AKI)
            prob = head(batch_treatment)
            outputs.append(prob)
            
        return torch.cat(outputs, dim=1) # [Batch, N_Side_Effects]
