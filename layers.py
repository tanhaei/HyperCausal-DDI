import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv

class CausalDeconfoundingLayer(nn.Module):
    """
    Implements the Neural Backdoor Adjustment (Section 3.4 of the paper).
    Estimates P(Y|do(T)) by stratifying over confounder prototypes.
    """
    def __init__(self, in_dim, n_prototypes=20, hidden_dim=64):
        super(CausalDeconfoundingLayer, self).__init__()
        self.n_prototypes = n_prototypes
        
        # Dictionary of confounder prototypes (C = {c1, ..., cK})
        # Initialized randomly, refined during training via gradients
        self.confounder_dict = nn.Parameter(torch.randn(n_prototypes, in_dim))
        
        # Attention mechanism to learn P(Z|T) proxy weights (Eq. 5)
        self.attention_W = nn.Linear(in_dim, in_dim)
        
        # Final predictor f(T, Z)
        self.predictor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Output specific side effect probability
        )

    def forward(self, treatment_emb):
        """
        Args:
            treatment_emb (T): Drug combination embeddings [Batch, Dim]
        Returns:
            interventional_prob: P(Y|do(T))
        """
        batch_size = treatment_emb.size(0)
        
        # 1. Calculate Attention Weights (alpha_k)
        # Query: Treatment, Key: Confounders
        query = self.attention_W(treatment_emb).unsqueeze(1) # [B, 1, D]
        keys = self.confounder_dict.unsqueeze(0).expand(batch_size, -1, -1) # [B, K, D]
        
        scores = torch.sum(query * keys, dim=-1) # [B, K]
        alpha_k = F.softmax(scores, dim=1) # Attention weights sum to 1
        
        # 2. Backdoor Adjustment (Eq. 5 approximation)
        # We compute f(T + c_k) for all k and weight them by alpha_k
        # Note: In the paper, we adjust features. Here we use addition/fusion.
        
        out_probs = 0
        for k in range(self.n_prototypes):
            c_k = self.confounder_dict[k].unsqueeze(0).expand(batch_size, -1)
            
            # Combine Treatment + Confounder (T, z)
            # Simple addition fusion as typically used in residual causal blocks
            combined_repr = treatment_emb + c_k 
            
            # Prediction f(T, z)
            pred_k = torch.sigmoid(self.predictor(combined_repr))
            
            # Weighted Sum: sum( alpha_k * f(T, c_k) )
            weight = alpha_k[:, k].unsqueeze(1)
            out_probs += weight * pred_k
            
        return out_probs

class SparseHypergraphLayer(nn.Module):
    """
    Wrapper for PyG's HypergraphConv to handle sparse inputs efficiently.
    Corresponds to Module 2 in Methodology.
    """
    def __init__(self, in_dim, out_dim):
        super(SparseHypergraphLayer, self).__init__()
        self.conv = HypergraphConv(in_dim, out_dim, use_attention=True)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x, hyperedge_index):
        x = self.conv(x, hyperedge_index)
        x = self.bn(x)
        return F.elu(x)
