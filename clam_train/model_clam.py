import torch
import torch.nn as nn
import torch.nn.functional as F

class Attn_Net_Gated(nn.Module):
    """
    Gated Attention Network
    """
    def __init__(self, L=1024, D=256, dropout=True, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A) 
        return A

class CLAM_SB(nn.Module):
    """
    CLAM Single Branch
    """
    def __init__(self, input_dim=384, hidden_dim=256, n_classes=2, k_sample=8, dropout=True):
        super(CLAM_SB, self).__init__()
        self.input_dim = input_dim
        self.k_sample = k_sample # Top-K / Bottom-K
        self.n_classes = n_classes
        
        self.fc1 = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.25)) if input_dim > 512 else nn.Identity()
        feat_dim = 512 if input_dim > 512 else input_dim

        self.attention_net = Attn_Net_Gated(L=feat_dim, D=hidden_dim, dropout=dropout, n_classes=1)
        
        self.classifiers = nn.Linear(feat_dim, n_classes)
        
        self.instance_classifiers = nn.Linear(feat_dim, 2) 

    def forward(self, h, label=None, instance_eval=False):
        
        h = self.fc1(h)
        
        A = self.attention_net(h)  
        A = torch.transpose(A, 1, 0)  
        A_prob = F.softmax(A, dim=1) 

        M = torch.mm(A_prob, h) 
        logits = self.classifiers(M) 
        
        instance_loss = torch.tensor(0.0).to(h.device)
        
        if self.training and label is not None:

            total_inst = h.size(0)
            k = min(self.k_sample, total_inst // 2) 
            
            top_k_indices = torch.topk(A, k, dim=1)[1].view(-1)

            bottom_k_indices = torch.topk(A, k, dim=1, largest=False)[1].view(-1)
            
            top_k_features = h.index_select(0, top_k_indices)
            bottom_k_features = h.index_select(0, bottom_k_indices)
            
            p_top = self.instance_classifiers(top_k_features)
            p_bottom = self.instance_classifiers(bottom_k_features)
            
            if label == 1:
                inst_labels_top = torch.ones(k, dtype=torch.long).to(h.device)
                inst_labels_bottom = torch.zeros(k, dtype=torch.long).to(h.device)
            
            else:
                inst_labels_top = torch.zeros(k, dtype=torch.long).to(h.device)
                inst_labels_bottom = torch.zeros(k, dtype=torch.long).to(h.device)
            
            loss_top = F.cross_entropy(p_top, inst_labels_top)
            loss_bottom = F.cross_entropy(p_bottom, inst_labels_bottom)
            instance_loss = (loss_top + loss_bottom) / 2

        return logits, A_prob, instance_loss
