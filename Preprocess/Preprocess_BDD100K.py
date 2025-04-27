#!/usr/bin/env python
# coding: utf-8

# In[1]:


import kagglehub

# Download latest version
path = kagglehub.dataset_download("robikscube/driving-video-with-object-tracking")

print("Path to dataset files:", path)


# In[1]:


import os
import json
import random
from glob import glob

#Constants
SOURCE_VIDEO_DIR = "BDD100K/videos"
OUTPUT_FRAMES_DIR = "BDD100K/BDD_Frames"
METADATA_OUTPUT_DIR = "Dataset"

#Prepare
os.makedirs(OUTPUT_FRAMES_DIR, exist_ok = True)

#Gather all video files
video_files = sorted(glob(os.path.join(SOURCE_VIDEO_DIR, "*.mov")))

#Shuffle and split
random.seed(42)
random.shuffle(video_files)
train_videos = video_files[:700]
val_videos = video_files[700:1000]

#Function to create metadata
def create_metadata_entry(video_id, num_frames, subset):
    return {
        "video_start": 0,
        "video_end": num_frames - 1,
        "anomaly_start": None,
        "anomaly_end": None,
        "anomaly_class": "normal",
        "num_frames": num_frames,
        "subset": subset
    }


# In[18]:


def videos_to_frames(video_path, output_dir, fps=10):
    import os
    import subprocess

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print(video_path + " has been processed!")

    # Use a list for the command (preferred and safer)
    query = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={fps}",
        os.path.join(output_dir, "%06d.jpg"),
        "-hide_banner",
        "-loglevel", "error"
    ]

    print("Running:", " ".join(query))
    result = subprocess.run(query, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print("ffmpeg failed:")
        print(result.stderr.decode())
    else:
        print("ffmpeg succeeded.")

    return len([f for f in os.listdir(output_dir) if f.endswith(".jpg")])


# In[19]:


#Prepare metadata dicts
metadata_train = {}
metadata_val = {}

# Simulate extracting frames and generating metadata
for video_path in train_videos:
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(OUTPUT_FRAMES_DIR, video_id)

    num_frames = videos_to_frames(video_path, output_dir, fps=10)
    metadata_train[video_id] = create_metadata_entry(video_id, num_frames, "train")
    
for video_path in val_videos:
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(OUTPUT_FRAMES_DIR, video_id)
    
    num_frames = videos_to_frames(video_path, output_dir, fps=10)
    metadata_val[video_id] = create_metadata_entry(video_id, num_frames, "val")
    
# Save JSONs
train_json_path = os.path.join(METADATA_OUTPUT_DIR, "metadata_train_bdd.json")
val_json_path = os.path.join(METADATA_OUTPUT_DIR, "metadata_val_bdd.json")
with open(train_json_path, "w") as f:
    json.dump(metadata_train, f, indent=2)

with open(val_json_path, "w") as f:
    json.dump(metadata_val, f, indent=2)

# Save train/val split files
train_split_path = os.path.join(METADATA_OUTPUT_DIR, "train_split_bdd.txt")
val_split_path = os.path.join(METADATA_OUTPUT_DIR, "val_split_bdd.txt")

with open(train_split_path, "w") as f:
    f.write("\n".join(metadata_train.keys()))

with open(val_split_path, "w") as f:
    f.write("\n".join(metadata_val.keys()))


# In[20]:


#Load existing metadata JSON files
existing_train_json_path = "dataset/metadata_train_preprocessed.json"
existing_val_json_path = "dataset/metadata_val_preprocessed.json"

with open(existing_train_json_path, "r") as f:
    existing_metadata_train = json.load(f)
with open(existing_val_json_path, "r") as f:
    existing_metadata_val = json.load(f)

#Load the new BDD metadata files created earlier
with open("dataset/metadata_train_bdd.json", "r") as f:
    metadata_train_bdd = json.load(f)
with open("dataset/metadata_val_bdd.json", "r") as f:
    metadata_val_bdd = json.load(f)

#Append BDD entries to existing metadata
existing_metadata_train.update(metadata_train_bdd)
existing_metadata_val.update(metadata_val_bdd)

#Save the merged metadata back
with open("dataset/metadata_train_merged.json", "w") as f:
    json.dump(existing_metadata_train, f, indent=2)

with open("dataset/metadata_val_merged.json", "w") as f:
    json.dump(existing_metadata_val, f, indent=2)


# In[21]:


#Merge the split txt files
with open("dataset/train_split_updated.txt", "r") as f:
    train_ids = f.read().splitlines()

with open("dataset/val_split_updated.txt", "r") as f:
    val_ids = f.read().splitlines()

with open("dataset/train_split_bdd.txt", "r") as f:
    bdd_train_ids = f.read().splitlines()

with open("dataset/val_split_bdd.txt", "r") as f:
    bdd_val_ids = f.read().splitlines()

# Merge and remove duplicates
merged_train_ids = sorted(set(train_ids + bdd_train_ids))
merged_val_ids = sorted(set(val_ids + bdd_val_ids))

# Save merged split files
with open("dataset/train_split_merged.txt", "w") as f:
    f.write("\n".join(merged_train_ids))

with open("dataset/val_split_merged.txt", "w") as f:
    f.write("\n".join(merged_val_ids))


# In[2]:


# Paths
train_split_path = "dataset/train_split_updated.txt"
val_split_path = "dataset/val_split_updated.txt"
dota_frames_dir = "DoTA/DoTA_Frames"

# Read both split files
with open(train_split_path, 'r') as f:
    train_ids = [line.strip() for line in f if line.strip()]

with open(val_split_path, 'r') as f:
    val_ids = [line.strip() for line in f if line.strip()]

# Merge them
all_ids = train_ids + val_ids

# Get all actual folder names under DoTA_Frames
existing_folders = set(os.listdir(dota_frames_dir))

# Keep only video IDs that have matching folders
valid_ids = [vid for vid in all_ids if vid in existing_folders]

print(f"Original IDs: {len(all_ids)}")
print(f"Valid IDs after checking: {len(valid_ids)}")


# In[3]:


# Save the filtered IDs into a new txt file
output_path = "dataset/combined_split_filtered.txt"
with open(output_path, 'w') as f:
    f.write('\n'.join(valid_ids))

print(f"Filtered split file saved to: {output_path}")


# In[4]:


import random

# Paths
filtered_dota_path = "dataset/combined_split_filtered.txt"
train_bdd_path = "dataset/train_split_bdd.txt"
val_bdd_path = "dataset/val_split_bdd.txt"

# Read all files
with open(filtered_dota_path, 'r') as f:
    dota_ids = [line.strip() for line in f if line.strip()]

with open(train_bdd_path, 'r') as f:
    bdd_train_ids = [line.strip() for line in f if line.strip()]

with open(val_bdd_path, 'r') as f:
    bdd_val_ids = [line.strip() for line in f if line.strip()]

# Combine DoTA and BDD IDs
all_ids = dota_ids + bdd_train_ids + bdd_val_ids

print(f"Total number of video IDs before shuffling: {len(all_ids)}")

# Random shuffle
random.seed(42)
random.shuffle(all_ids)

# Split into 80% train, 20% val
split_idx = int(0.8 * len(all_ids))
train_ids = all_ids[:split_idx]
val_ids = all_ids[split_idx:]

print(f"Train set size: {len(train_ids)}")
print(f"Validation set size: {len(val_ids)}")


# In[5]:


# Output paths
final_train_split_path = "dataset/final_train_split.txt"
final_val_split_path = "dataset/final_val_split.txt"

# Save train and val splits
with open(final_train_split_path, 'w') as f:
    f.write('\n'.join(train_ids))

with open(final_val_split_path, 'w') as f:
    f.write('\n'.join(val_ids))

print(f"Final split files saved: {final_train_split_path}, {final_val_split_path}")


# In[6]:


import json

# Paths
metadata_train_path = "dataset/metadata_train_merged.json"
metadata_val_path = "dataset/metadata_val_merged.json"
final_train_split_path = "dataset/final_train_split.txt"
final_val_split_path = "dataset/final_val_split.txt"

# Output paths
final_metadata_train_path = "dataset/final_metadata_train.json"
final_metadata_val_path = "dataset/final_metadata_val.json"

# Load metadata files
with open(metadata_train_path, 'r') as f:
    metadata_train = json.load(f)

with open(metadata_val_path, 'r') as f:
    metadata_val = json.load(f)

# Merge both metadata dictionaries
metadata_all = {**metadata_train, **metadata_val}

# Load final split IDs
with open(final_train_split_path, 'r') as f:
    final_train_ids = [line.strip() for line in f if line.strip()]

with open(final_val_split_path, 'r') as f:
    final_val_ids = [line.strip() for line in f if line.strip()]

# Filter metadata
final_metadata_train = {k: v for k, v in metadata_all.items() if k in final_train_ids}
final_metadata_val = {k: v for k, v in metadata_all.items() if k in final_val_ids}

print(f"Original metadata entries: {len(metadata_all)}")
print(f"Filtered Train Metadata Entries: {len(final_metadata_train)}")
print(f"Filtered Val Metadata Entries: {len(final_metadata_val)}")


# In[7]:


# Save new metadata files
with open(final_metadata_train_path, 'w') as f:
    json.dump(final_metadata_train, f, indent=2)

with open(final_metadata_val_path, 'w') as f:
    json.dump(final_metadata_val, f, indent=2)

print(f"Saved new filtered metadata files to:")
print(f" - {final_metadata_train_path}")
print(f" - {final_metadata_val_path}")


# In[ ]:




