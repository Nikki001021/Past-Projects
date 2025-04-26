#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import wandb

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random


# In[3]:


#0. Decompress Datasets
#!bash prepare_data2.sh


# In[2]:


from torchvision.transforms.autoaugment import AutoAugment, AutoAugmentPolicy
#1. Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
])

#2. Feature Extractor: ResNet
class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.output_dim = resnet.fc.in_features
        
    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x).squeeze(-1).squeeze(-1)
        return features


# In[3]:


#3. BiLSTM Classifier
class VideoClassifier(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim,
                                           num_layers=num_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        # hn shape: (num_layers * num_directions, batch_size, hidden_dim)
        # We need to concatenate the forward and backward hidden states
        hn = torch.cat((hn[-2], hn[-1]), dim=1)  # concat last two hidden states
        out = self.classifier(hn)
        return out.squeeze()


# In[4]:


#4. Dataset Class
class VideoFrameDataset(Dataset):
    def __init__(self, split_file, metadata_file, dota_dir, bdd_dir, sample_count=64):
        with open(split_file, 'r') as f:
            self.video_ids =  [line.strip() for line in f if line.strip()]
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
            
        self.dota_dir = dota_dir
        self.bdd_dir = bdd_dir
        self.sample_count = sample_count
        
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        label = 0 if self.metadata[video_id]["anomaly_class"] == "normal" else 1
    
        if label == 0:
            folder = os.path.join(self.bdd_dir, video_id)
        else:
            folder = os.path.join(self.dota_dir, video_id, "images")
            
        if not os.path.exists(folder):
            print(f"[WARNING] Missing folder: {folder} — skipping.")
            # Return a dummy result that will be ignored (skip sample logic in loader)
            return self.__getitem__((idx + 1) % len(self))
    
        frame_paths = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith('.jpg') and not f.startswith('._')
        ])

        total = len(frame_paths)
    
        if total < self.sample_count:
            frame_paths += [frame_paths[-1]] * (self.sample_count - total)
            selected = frame_paths
        else:
            step = total / self.sample_count
            selected = [frame_paths[int(i * step)] for i in range(self.sample_count)]

        images = [transform(Image.open(f).convert('RGB')) for f in selected]
        return torch.stack(images), torch.tensor(label, dtype=torch.float32)


# In[5]:


def evaluate(model, loader, feature_extractor, device, criterion=None):
    model.eval()
    preds = []
    targets = []
    total_loss = 0
    with torch.no_grad():
        for frames, label in loader:
            frames = frames.to(device)
            label = label.to(device)

            batch_size, seq_len, C, H, W = frames.shape
            frames = frames.view(batch_size * seq_len, C, H, W)

            features = feature_extractor(frames)
            features = features.view(batch_size, seq_len, -1)

            output = model(features)
            preds += (output > 0.5).int().tolist()
            targets += label.int().tolist()

            # Compute loss if criterion is provided
            if criterion is not None:
                loss = criterion(output, label.unsqueeze(0) if output.dim() == 0 else label)
                total_loss += loss.item()

    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds)
    rec = recall_score(targets, preds)
    f1 = f1_score(targets, preds)

    avg_loss = total_loss / len(loader) if criterion else None

    return acc, prec, rec, f1, avg_loss


# In[6]:


#6. Training Loop
def train(model, train_loader, val_loader, feature_extractor, device, epochs):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    
    best_val_f1 = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for frames, label in train_loader:
            frames = frames.to(device)  # [batch, seq, 3, 224, 224]
            label = label.to(device)
    
            batch_size, seq_len, C, H, W = frames.shape
            frames = frames.view(batch_size * seq_len, C, H, W)  # flatten

            features = feature_extractor(frames)  # [batch*seq, feature_dim]
            features = features.view(batch_size, seq_len, -1)  # reshape back

            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            
        avg_train_loss = running_loss / len(train_loader)
        # Evaluate on train and val sets
        train_acc, train_prec, train_rec, train_f1, train_eval_loss = evaluate(model, train_loader, feature_extractor, device, criterion)
        val_acc, val_prec, val_rec, val_f1, val_eval_loss = evaluate(model, val_loader, feature_extractor, device, criterion)

        # Print nicely
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} | Eval Loss: {train_eval_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_eval_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}\n")

        # 🪄 Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_eval_loss": train_eval_loss,
            "val_loss": val_eval_loss,
            "train_accuracy": train_acc,
            "train_precision": train_prec,
            "train_recall": train_rec,
            "train_f1": train_f1,
            "val_accuracy": val_acc,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "val_f1": val_f1,
        })
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'feature_extractor_state_dict': feature_extractor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1
            }, "best_model.pth")
            print(f"Saved new best model at epoch {epoch+1} with Val F1: {val_f1:.4f}")


# In[ ]:


#7. Inference
if __name__ == '__main__':
    #Paths
    train_split_file = "dataset/final_train_split.txt"
    val_split_file = "dataset/final_val_split.txt"
    train_metadata_file = "dataset/final_metadata_train.json"
    val_metadata_file = "dataset/final_metadata_val.json"
    dota_dir = "DoTA/DoTA_Frames"
    bdd_dir = "BDD100K/BDD_Frames"
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_dataset = VideoFrameDataset(train_split_file, train_metadata_file, dota_dir, bdd_dir)
    val_dataset = VideoFrameDataset(val_split_file, val_metadata_file, dota_dir, bdd_dir)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8, 
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=8,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # Models
    feature_extractor = ResNetFeatureExtractor().to(device)
    classifier = VideoClassifier(feature_dim=feature_extractor.output_dim)
    
    wandb.init(
        project="car-crash-detection",
        name="resnet-lstm-run3",
        config={
            "epochs": 35,
            "batch_size": 8,
            "learning_rate": 1e-4,
            "model": "ResNet18 + LSTM",
            "dataset": "DoTA + BDD100K Combined",
        }
    )
    
    
    # Train
    train(classifier, train_loader, val_loader, feature_extractor, device, epochs=35)


# In[ ]:




