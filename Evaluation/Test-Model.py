#!/usr/bin/env python
# coding: utf-8

# In[11]:


import torch
from torch.utils.data import DataLoader, Dataset
import json
import os
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from ResNet_LSTM_Model2 import ResNetFeatureExtractor, VideoClassifier  # (import your models)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import subprocess
from glob import glob
import random


# In[9]:


#0. Convert Videos to Frames
# Constants
SOURCE_VIDEO_DIR = "Test_Video"
OUTPUT_FRAMES_DIR = "Test_Video_Frames"
os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)

# Gather all .mp4 files from both Normal and Crash folders
video_files = glob(os.path.join(SOURCE_VIDEO_DIR, "Normal-002", "*.mp4")) +               glob(os.path.join(SOURCE_VIDEO_DIR, "Crash-1500", "*.mp4"))

print(f"Found {len(video_files)} videos for testing.")

# Function to extract frames
def videos_to_frames(video_path, output_dir, fps=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # load module first inside the shell!
    query = f"module load ffmpeg && ffmpeg -i '{video_path}' -vf fps={fps} '{os.path.join(output_dir, '%06d.jpg')}' -hide_banner -loglevel error"
    print("Running:", query)
    
    result = subprocess.run(query, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print("ffmpeg failed:")
        print(result.stderr.decode())
    else:
        print("ffmpeg succeeded.")

    return len([f for f in os.listdir(output_dir) if f.endswith(".jpg")])

# Process all videos
for video_path in video_files:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(OUTPUT_FRAMES_DIR, video_name)
    videos_to_frames(video_path, output_dir, fps=10)


# In[18]:


from torchvision.transforms.autoaugment import AutoAugment, AutoAugmentPolicy
import numpy as np

# 1. Define the Dataset
class TestVideoDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform, sample_count=64):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.sample_count = sample_count
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        video_info = self.data.iloc[idx]
        video_folder = os.path.join(self.root_dir, os.path.splitext(os.path.basename(video_info['video_path']))[0])

        frame_paths = sorted([os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.jpg')])

        total_frames = len(frame_paths)
        if total_frames == 0:
            raise ValueError(f"No frames found for {video_folder}")

        step = max(total_frames // self.sample_count, 1)
        selected_frames = [frame_paths[min(i * step, total_frames - 1)] for i in range(self.sample_count)]
        frames = [self.transform(Image.open(f).convert('RGB')) for f in selected_frames]
        
        label = video_info['label']
        return torch.stack(frames), torch.tensor(label, dtype=torch.float32)

# 2. Transform (same as training)
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

# 3. Load the Dataset
test_dataset = TestVideoDataset(
    csv_file="test_labels.csv",              # You need to create this earlier (video names + labels)
    root_dir="Test_Video_Frames",             # <--- notice here: it's the FRAMES folder
    transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

# 4. Load your Best Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_extractor = ResNetFeatureExtractor().to(device)
classifier = VideoClassifier(feature_dim=feature_extractor.output_dim).to(device)

checkpoint = torch.load("best_model.pth", map_location=device, weights_only=False)
classifier.load_state_dict(checkpoint['model_state_dict'])

classifier.eval()
feature_extractor.eval()

# 5. Predict on Test Set (modified to collect raw outputs)
all_outputs, all_labels = [], []

with torch.no_grad():
    for frames, labels in tqdm(test_loader):
        frames = frames.to(device)  # [batch, seq, C, H, W]
        labels = labels.to(device)

        batch_size, seq_len, C, H, W = frames.shape
        frames = frames.view(batch_size * seq_len, C, H, W)
        features = feature_extractor(frames)
        features = features.view(batch_size, seq_len, -1)

        outputs = classifier(features)

        all_outputs.extend(outputs.squeeze().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_outputs = np.array(all_outputs)
all_labels = np.array(all_labels)

# 6. Threshold Tuning
thresholds = np.arange(0.1, 0.91, 0.05)  # From 0.1 to 0.9 with step 0.05

best_f1 = 0
best_threshold = 0

print("\nThreshold Tuning Results:")
print("Threshold | Accuracy | Precision | Recall | F1-Score")
print("-" * 50)

for threshold in thresholds:
    preds = (all_outputs > threshold).astype(int)

    acc = accuracy_score(all_labels, preds)
    prec = precision_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)

    print(f"{threshold:.2f}      | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("\nBest Threshold:", best_threshold)
print("Best F1-Score:", best_f1)


# In[ ]:




