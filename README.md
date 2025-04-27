# DS542 Final Project: AI-Based Car Crash Detection Using Deep Learning
### Project Description:
Car crash detection is a critical application of AI-driven traffic surveillance, enabling accident identification and response. This project aims to develop a deep learning-based system that detects car crashes from video data. We will be using the pre-trained Yolov8 model to integrate computer vision techniques for video analysis. We will train the model on DoTA dataset (traffic accident videos) and BDD100K dataset (normal driving videos). The final system will be evaluated based on precision and recall.

### Teammate Information:
1. Shiyi Chen, shiychen@bu.edu
2. Yuanchen Yin, yinthyg@bu.edu
3. Yuchen Li, nikkili@bu.edu

### Datasets:
1. **[Detection of Traffic Anomaly (DoTA)](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly/tree/master):** includes 4,677 videos with annotated anomalies (we will only focus on car-crashed related footages)
2. **[Normal Driving Footages (Part of BDD100K dataset)](https://www.kaggle.com/datasets/robikscube/driving-video-with-object-tracking/data):** Provides normal driving videos for comparison to get a more comprehensive insight on judging potential car crashes
3. **[Car Crash Dataset](https://github.com/Cogito2012/CarCrashDataset):** Provides 3000 normal driving videos and 1500 crash accident videos as test dataset for model evaluation

### Navigation:
For **data preprocessing**, see the **"Preproess"** folder, detailed preprocessing steps are listed in "README" under "Preprocess" folder<br>
For all the **metadata** files, see the json files with "metadata" under **"dataset"** folder
