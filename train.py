import torch
import torch.nn as nn
import torch.optim as optim
from model import HyperCausalDDI

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in side effect prediction.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

def train():
    # --- Hyperparameters ---
    N_DRUGS = 1245
    HIDDEN_DIM = 128
    N_SIDE_EFFECTS = 65
    LR = 1e-4
    EPOCHS = 50
    
    # --- Device Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- Initialize Model ---
    model = HyperCausalDDI(
        n_drugs=N_DRUGS,
        str_input_dim=64, # Dummy
        sem_input_dim=768,
        hidden_dim=HIDDEN_DIM,
        n_side_effects=N_SIDE_EFFECTS
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = FocalLoss(gamma=2.0)
    
    # --- Dummy Data Generator (Replace with MIMIC-IV Data Loader) ---
    # In reality, load your sparse hyperedge_index from processed CSVs
    print("Generating dummy data for simulation...")
    num_hyperedges = 5000 # Total prescriptions
    
    # Create random hypergraph connections (node_idx, hyperedge_idx)
    nodes = torch.randint(0, N_DRUGS, (num_hyperedges * 3,)) # Avg 3 drugs per Rx
    edges = torch.repeat_interleave(torch.arange(num_hyperedges), 3)
    hyperedge_index = torch.stack([nodes, edges], dim=0).to(device)
    
    # Labels: [Num_Hyperedges, N_Side_Effects]
    labels = torch.randint(0, 2, (num_hyperedges, N_SIDE_EFFECTS)).float().to(device)
    
    # --- Training Loop ---
    model.train()
    print("Starting training...")
    
    batch_size = 64
    num_batches = num_hyperedges // batch_size
    
    for epoch in range(EPOCHS):
        total_loss = 0
        perm = torch.randperm(num_hyperedges)
        
        for i in range(num_batches):
            idx = perm[i*batch_size : (i+1)*batch_size]
            
            optimizer.zero_grad()
            
            # Forward pass
            # Note: We pass the full hypergraph structure but slice output for batch
            preds = model(
                structure_data=None, # Passed None for dummy run
                semantic_emb=None, 
                hyperedge_index=hyperedge_index,
                batch_indices=idx.to(device)
            )
            
            loss = criterion(preds, labels[idx])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/num_batches:.4f}")

if __name__ == "__main__":
    train()
