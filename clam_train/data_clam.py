import os
import torch
import pandas as pd
from torch.utils.data import Dataset

class WsiFeatureDataset(Dataset):
    def __init__(self, csv_path, feature_dir):
        self.df = pd.read_csv(csv_path)
        self.feature_dir = feature_dir
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = row['slide_id']
        label = int(row['label'])
        
        pt_path = os.path.join(self.feature_dir, f"{slide_id}.pt")
        
        try:
            data = torch.load(pt_path)
            if isinstance(data, dict):
                features = data['features'] 
            else:
                features = data 
            
            if not isinstance(features, torch.FloatTensor):
                 features = features.float()
                 
        except Exception as e:
            print(f"Error loading {pt_path}: {e}")
            features = torch.zeros(1, 384) 

        return features, label

def collate_mil(batch):
    img = batch[0][0]
    label = torch.tensor(batch[0][1]).long()
    return img, label

