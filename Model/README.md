# Model Walkthrough
### 1. Data Preparation

For this model, we used the following datasets:
1. **DoTA (Detection of Traffic Anomalies)** for crash video frames.
2. **BDD100K (Berkeley DeepDrive)** for normal driving video frames.

If you want to skip the video preprocess steps, we could refer to the following links for downloading compressed ready-to-go video frames:
1. For DoTA: https://www.dropbox.com/scl/fo/ipghvnyhei90vy6hpif9t/AFFryPMJ-bzY7C2ukOXWhHY?rlkey=az0dugb1aerzjyxciahwn6p2j&st=d8pjfwot&dl=0
2. For BDD100K: https://www.dropbox.com/scl/fi/8nj6s9hg0e2v9b5av9bzb/BDD_Frames.tar.gz?rlkey=osk9dhokl938eurmtucjkvnkr&st=ka0zrwl6&dl=0<br><br>
To decompress all the datasets, please refer to ```prepare_data2.sh``` which:
- Combines the ```.tar.gz``` part files into complete archives
- Extracts the archives into correct subfolders: ```DoTA/DoTA_Frames/``` and ```BDD100K/BDD_Frames/```. If you don't have the folders set up already, please make sure to change the following based on your own working directory:
```
mkdir -p "${WORK_DIR}/DoTA/DoTA_Frames"
mkdir -p "${WORK_DIR}/BDD100K/BDD_Frames"
```
- Automatically organizes all frames based on the dataset
Inside ```ResNet_LSTM_Model2```, there is already built-in code for calling ```prepare_data2.sh```. If you want to call your own decompression file, please refer to the following code:
```
!bash [name of your .sh file]
```

### 2. Model Architecture
This project uses ```ResNet18``` with ```BiLSTM``` for detection. Here is a walkthrough for the model architecture:<br><br>
**1. Image Preprocessing:**
- Each image is resized to ```(224, 224)```
- Apply aggressive augmentations: random cropping, flipping, perspective distortion, color jittering, Gaussian blur, AutoAugment, and random erasing
- Normalize with ```ImageNet``` standards

**2. Feature Extraction:**
- A **pretrained ```ResNet-18```** is used, with the fully-connected (FC) layer removed
- The output is a **512-dimensional** feature vector per frame

**3. Sequence Modeling**
- A **```BiLSTM``` (Bidirectional Long Short-Term Memory)** network processes the sequence of frame features in **both forward and backward directions**, allowing the model to capture both past and future frame dependencies
- Takes in 64 frames per video clip, outputs a video-level representation

**4. Classification**
- A small fully connected (linear) layer + sigmoid activation classifies whether the video sequence is a **car crash (1)** or **normal (0)**

### 3. Training Strategy
- **Loss Function:** Binary Cross-Entropy Loss (BCE Loss)
- **Optimizer:** **Adam** with a learning rate of ```1e-4```
- **Batch Size:** 8 sequences per batch
- **Logging:** Training progress, metrics, and loss values are tracked with Weights & Biases **(wandb)**
- **Best Model Saving:** We save the model checkpoint whenever **validation F1-Score** improves. Refer to [this link](https://www.dropbox.com/scl/fi/qksu11abxyb2twa9tgk22/best_model.pth?rlkey=z4x840us19b2iulypkv6rr7bd&st=ix85yh6p&dl=0) to download our best model

### 4. Setup Instructions
To run ```ResNet_LSTM_Model2.py```, you need to import following library:
```
os
json
torch
torchvision
PIL
sklearn
random
wandb
```
