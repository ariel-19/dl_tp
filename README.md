# Deep Learning – Practical Works (TP 1 to TP 5)

**Author:** Shey Cyrille Njeta  

This repository contains the **source code**, **Docker configurations**, and **reports** related to the Deep Learning practical works, covering **TP 1, 2, 3, 4, and 5**.

---

## 🛠 Installation and Setup

### 1. Clone the repository
```bash
git clone <repository-url>
cd <repository-folder>
2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

📂 Execution Guide

This repository follows a structured and progressive learning path in Deep Learning.
The goal is to start with fundamental models and gradually introduce advanced techniques, ranging from deployment to computer vision and medical data processing.

✅ TP 1: Fundamentals and Deployment

Focus: Basic model training, API serving, and Docker containerization.

In this first practical work, a simple handwritten digit recognition model (MNIST) is trained, exposed through a REST API, and deployed using Docker to ensure portability and reproducibility.

Scripts

train_model.py
Trains a baseline MNIST model and saves it as mnist_model.h5.

python3 train_model.py


app.py
Flask API used to serve predictions from the trained model.

python3 app.py


(Server runs on port 5000)

Docker
docker build -t mnist-app .
docker run -p 5000:5000 mnist-app

✅ TP 2: Improving Neural Networks

Focus: Regularization, optimization, and experiment tracking.

This practical introduces techniques to improve model generalization, including bias/variance analysis and regularization methods.
MLflow is used to track and compare experiments efficiently.

Script

train_model.py

Train / Validation / Test split

Regularization (L2, Dropout, Batch Normalization)

Optimizer comparison

Experiment tracking with MLflow

python3 train_model.py

✅ TP 3: Computer Vision and CNNs

Focus: Convolutional Neural Networks, ResNet, and Style Transfer.

This practical focuses on image analysis using CNNs.
It also explores neural style transfer, demonstrating how deep networks can separate image content from visual style.

Scripts

cnn_classification.py
Trains a basic CNN and a ResNet model on the CIFAR-10 dataset.

python3 cnn_classification.py


style_transfer.py
Implements neural style transfer using the VGG16 network.

python3 style_transfer.py

✅ TP 4: Segmentation and 3D Data

Focus: Image segmentation and 3D convolutions.

This practical addresses high-precision tasks, particularly relevant to medical imaging.
A U-Net architecture is used for image segmentation, while 3D convolutions are applied to volumetric data.

Scripts

unet_segmentation.py
Trains a U-Net model using Dice and IoU metrics.

python3 unet_segmentation.py


conv3d_demo.py
Implements a Conv3D block and logs the architecture using MLflow.

python3 conv3d_demo.py

✅ TP 5: Advanced Approaches and Integration

Focus: Consolidation and advanced applications.

TP 5 aims to consolidate all concepts introduced in previous practical works.
It emphasizes the integration of advanced techniques, adaptation to realistic use cases, and comprehensive evaluation of the developed models, particularly in a medical context.

📌 Note

Each practical work is designed to be independent yet complementary, enabling a gradual skill progression in Deep Learning—from foundational concepts to advanced real-world applications.


---

If you want, I can also:
- 📘 make a **short academic version**
- 🧪 add a **Results / Experiments** section
- 🐳 improve **Docker + MLflow documentation**
- 🌍 adapt it for **GitHub classroom or course submission**

Just tell me 😊
