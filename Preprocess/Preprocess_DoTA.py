#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os


# In[7]:


#Load the original metadata files
with open("dataset/metadata_train.json", "r") as f:
    metadata_train = json.load(f)
with open("dataset/metadata_val.json", "r") as f:
    metadata_val = json.load(f)


# In[11]:


#Define anomaly classes to remove
exclude_keywords = [
    "leave_to_right", "leave_to_left", "unknown", "obstacle", "pedestrian"
]

def filter_metadata(metadata):
    filtered_metadata = {}
    for clip_id, data in metadata.items():
        anomaly_class = data.get("anomaly_class", "")
        #Check for exclusion by keyword
        if any (ex_kw in anomaly_class for ex_kw in exclude_keywords):
            continue
        #Keep the rest
        else:
            #Remove the prefix "ego" and "other" and strip extra whitespace
            cleaned_class = anomaly_class.split(": ", 1)[-1].strip()
            data["anomaly_class"] = cleaned_class
            filtered_metadata[clip_id] = data
    return filtered_metadata

#Filter both metadata sets
filtered_train = filter_metadata(metadata_train)
filtered_val = filter_metadata(metadata_val)


# In[12]:


#Save filtered metadata
with open ("dataset/metadata_train_preprocessed.json", "w") as f:
    json.dump(filtered_train, f, indent=2)
with open ("dataset/metadata_val_preprocessed.json", "w") as f:
    json.dump(filtered_val, f, indent=2)


# In[18]:


from collections import defaultdict

#Group the filtered metadata files and count the number of data for each anomaly class
def count_by_anomaly_class(metadata, name=""):
    class_counts = defaultdict(int)
    for clip in metadata.values():
        class_name = clip.get("anomaly_class", "unknown")
        class_counts[class_name] += 1
    print(f"\nAnomaly class distribution in {name}:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"{cls}: {count}")

count_by_anomaly_class(filtered_train, "filtered_train")
count_by_anomaly_class(filtered_val, "filtered_val")


# In[14]:


#Extract remaining video IDs to update split files
remaining_train_ids = set(filtered_train.keys())
remaining_val_ids = set(filtered_val.keys())

#Load and filter split files
def update_split_file(split_path, valid_ids):
    with open(split_path, "r") as f:
        lines = f.read().splitlines()
    return [line for line in lines if line in valid_ids]

updated_train_split = update_split_file("dataset/train_split.txt", remaining_train_ids)
updated_val_split = update_split_file("dataset/val_split.txt", remaining_val_ids)

#Save updated split files
with open("dataset/train_split_updated.txt", "w") as f:
    f.write("\n".join(updated_train_split))
with open("dataset/val_split_updated.txt", "w") as f:
    f.write("\n".join(updated_val_split))


# In[16]:


print(len(updated_train_split))
print(len(updated_val_split))


# In[ ]:




