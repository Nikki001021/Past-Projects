#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import random
import pandas as pd

def rename_videos(folder_path, prefix):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.mp4')])
    
    for idx, filename in enumerate(files):
        old_path = os.path.join(folder_path, filename)
        new_filename = f"{prefix}_{idx+1:06d}.mp4"  # zero-padded
        new_path = os.path.join(folder_path, new_filename)
        os.rename(old_path, new_path)
        
    print(f"Renamed {len(files)} videos in {folder_path} with prefix '{prefix}'.")

# Step 1: Rename videos
normal_dir = "Test_Video/Normal-002"
crash_dir = "Test_Video/Crash-1500"

rename_videos(normal_dir, "normal")
rename_videos(crash_dir, "crash")

# Step 2: Build a list of all videos + labels
normal_videos = [(os.path.join('Normal-002', f), 0) for f in os.listdir(normal_dir) if f.endswith('.mp4')]
crash_videos = [(os.path.join('Crash-1500', f), 1) for f in os.listdir(crash_dir) if f.endswith('.mp4')]

all_test_data = normal_videos + crash_videos

# Step 3: Shuffle the combined list
random.seed(42)  # reproducibility
random.shuffle(all_test_data)

# Step 4: Save to CSV
df = pd.DataFrame(all_test_data, columns=["video_path", "label"])
df.to_csv("test_labels.csv", index=False)

print(f"Saved shuffled test_labels.csv with {len(df)} entries!")


# In[ ]:




