import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader  # For hypergraph batches
from sklearn.metrics import roc_auc_score, average_precision_score
from model import HyperCausalDDI
import numpy as np
import os
from tqdm import tqdm  # Progress bar

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in side effect prediction.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    Improved: Added class weights for multi-label.
    """
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=65):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha] * num_classes).cuda() if torch.cuda.is_available() else torch.tensor([alpha] * num_classes)
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')  # Use logits for stability

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

def evaluate(model, loader, device, criterion):
    """Compute AUC/AUPR on validation set."""
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch.structure_data, batch.semantic_emb, batch.hyperedge_index, batch.batch)
            loss = criterion(preds, batch.y)
            total_loss += loss.item()
            
            all_preds.append(torch.sigmoid(preds).cpu())
            all_preds.append(batch.y.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    auc = roc_auc_score(all_labels, all_preds, average='macro')
    aupr = average_precision_score(all_labels, all_preds, average='macro')
    return total_loss / len(loader), auc, aupr

def train():
    # --- Hyperparameters ---
    N_DRUGS = 1245
    STR_INPUT_DIM = 64  # Atom features from RDKit
    SEM_INPUT_DIM = 768  # BioBERT
    HIDDEN_DIM = 128
    N_SIDE_EFFECTS = 65
    LR = 1e-4
    EPOCHS = 50
    BATCH_SIZE = 256  # Increased for efficiency
    PATIENCE = 5  # Early stopping
    
    # --- Device Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- Load Data (from data_preprocess) ---
    train_data = torch.load('processed/hypergraph_train.pt')
    val_data = torch.load('processed/hypergraph_val.pt')
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- Initialize Model ---
    model = HyperCausalDDI(
        n_drugs=N_DRUGS,
        str_input_dim=STR_INPUT_DIM,
        sem_input_dim=SEM_INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        n_side_effects=N_SIDE_EFFECTS
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = FocalLoss(gamma=2.0, num_classes=N_SIDE_EFFECTS).to(device)
    
    # --- Training Loop with Early Stopping ---
    best_val_loss = float('inf')
    patience_counter = 0
    os.makedirs('checkpoints', exist_ok=True)
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            preds = model(
                batch.structure_data if hasattr(batch, 'structure_data') else None,
                batch.semantic_emb if hasattr(batch, 'semantic_emb') else None,
                batch.hyperedge_index,
                batch.batch
            )
            loss = criterion(preds, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validate
        val_loss, val_auc, val_aupr = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, AUC={val_auc:.3f}, AUPR={val_aupr:.3f}")
        
        # Early stopping & Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print("Training complete! Best model saved to checkpoints/best_model.pth")

if __name__ == "__main__":
    train()