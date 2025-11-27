import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from model_clam import CLAM_SB
from data_clam import WsiFeatureDataset, collate_mil

def train_epoch(model, loader, optimizer, criterion, device, w_inst=0.7):
    model.train()
    total_loss = 0.
    total_inst_loss = 0.
    
    for features, label in loader:
        features = features.to(device) # Shape: [N, Dim]
        label = label.to(device).unsqueeze(0) # Shape: [1]
        
        optimizer.zero_grad()
        
        # Forward
        logits, _, instance_loss = model(features, label=label.item())
        
        # Calculate Loss
        slide_loss = criterion(logits, label)
        
        # CLAM Total Loss = Slide Loss + c * Instance Loss
        loss = slide_loss + w_inst * instance_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_inst_loss += instance_loss.item()
        
    return total_loss / len(loader), total_inst_loss / len(loader)

def eval_epoch(model, loader, device):
    model.eval()
    probs = []
    labels = []
    
    with torch.no_grad():
        for features, label in loader:
            features = features.to(device)
            
            # No need to calculate Instance Loss during Eval
            logits, _, _ = model(features, label=None)
            prob = F.softmax(logits, dim=1)[:, 1] # Get probability for label=1 (Tumor)
            
            probs.extend(prob.cpu().numpy())
            labels.append(label.item())
            
    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0
    return auc

# ==========================================
# 4. Main Program
# ==========================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='Path to dataset.csv')
    parser.add_argument('--feat_dir', type=str, required=True, help='Directory containing .pt files')
    parser.add_argument('--input_dim', type=int, default=384, help='ViT=384, ResNet=1024')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--w_inst', type=float, default=0.7, help='Weight for instance loss')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    # Device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    # Dataset
    dataset = WsiFeatureDataset(args.csv, args.feat_dir)
    # Simple Train/Val split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # DataLoader (BatchSize must be 1)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=collate_mil)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_mil)

    # Model Initialization
    model = CLAM_SB(input_dim=args.input_dim, n_classes=2, k_sample=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    print(f"📋 Training on {len(train_set)} slides, Validating on {len(val_set)} slides.")

    # Training Loop
    best_auc = 0.
    for epoch in range(args.epochs):
        train_loss, inst_loss = train_epoch(model, train_loader, optimizer, criterion, device, args.w_inst)
        val_auc = eval_epoch(model, val_loader, device)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Loss: {train_loss:.4f} (Inst: {inst_loss:.4f}) | "
              f"Val AUC: {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'best_clam_model.pth')
            print("  --> 💾 Model Saved!")
            
    print(f"🎉 Done! Best AUC: {best_auc:.4f}")