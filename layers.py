import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv  # pip install torch-geometric

class CausalDeconfoundingLayer(nn.Module):
    """
    Implements the Neural Backdoor Adjustment (Section 3.4 of the paper).
    Estimates P(Y|do(T)) by stratifying over confounder prototypes.
    Improved: Vectorized for efficiency (no loop over K).
    """
    def __init__(self, in_dim, n_prototypes=20, hidden_dim=64):
        super(CausalDeconfoundingLayer, self).__init__()
        self.n_prototypes = n_prototypes
        
        # Dictionary of confounder prototypes (C = {c1, ..., cK})
        # Pre-load from K-Means++ centroids if available
        self.confounder_dict = nn.Parameter(torch.randn(n_prototypes, in_dim))
        
        # Attention mechanism to learn P(Z|T) proxy weights (Eq. 5)
        self.attention_W = nn.Linear(in_dim, in_dim)
        
        # Final predictor f(T, Z)
        self.predictor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Added dropout for regularization
            nn.Linear(hidden_dim, 1) # Output specific side effect probability
        )

    def forward(self, treatment_emb):
        """
        Args:
            treatment_emb (T): Drug combination embeddings [Batch, Dim]
        Returns:
            interventional_prob: P(Y|do(T)) [Batch, 1]
        """
        batch_size = treatment_emb.size(0)
        
        # 1. Calculate Attention Weights (alpha_k) - Vectorized
        query = self.attention_W(treatment_emb).unsqueeze(1)  # [B, 1, D]
        keys = self.confounder_dict.unsqueeze(0).expand(batch_size, -1, -1)  # [B, K, D]
        scores = torch.sum(query * keys, dim=-1) / np.sqrt(treatment_emb.size(-1))  # Scaled dot-product [B, K]
        alpha_k = F.softmax(scores, dim=1)  # [B, K]
        
        # 2. Backdoor Adjustment (Eq. 5) - Vectorized
        combined_repr = treatment_emb.unsqueeze(1) + self.confounder_dict.unsqueeze(0)  # [B, K, D]
        pred_k = torch.sigmoid(self.predictor(combined_repr))  # [B, K, 1]
        out_probs = torch.sum(alpha_k.unsqueeze(-1) * pred_k, dim=1)  # Weighted sum [B, 1]
        
        return out_probs

class SparseHypergraphLayer(nn.Module):
    """
    Wrapper for PyG's HypergraphConv to handle sparse inputs efficiently.
    Corresponds to Module 2 in Methodology. Added dropout and residual.
    """
    def __init__(self, in_dim, out_dim):
        super(SparseHypergraphLayer, self).__init__()
        self.conv = HypergraphConv(in_dim, out_dim, use_attention=True)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, hyperedge_index):
        residual = x  # Residual connection
        x = self.conv(x, hyperedge_index)
        x = self.bn(x)
        x = self.dropout(F.elu(x))
        if residual.size(-1) == x.size(-1):  # Match dims for residual
            x = x + residual
        return x