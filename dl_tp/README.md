# Deep Learning Practical Works (TP 1, 2, 3, & 4)

**Author:** Shey Cyrille Njeta 


This repository contains the source code, Docker configurations, and reports for the Deep Learning practical assignments (TP 1 to 4).

## 🛠 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>

2. **Create and Activate a Virtual Environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

## 📂 Execution Guide

Think of this repository as a **progressive workshop**. We start by building a basic tool and putting it in a portable toolbox (Docker), then move to fine-tuning the engine, and finally build advanced visual sensors.

### ✅ Practical 1: Foundations & Deployment
*Focus: Base model training, API serving, and Docker containerization.*

In this step, we create a basic "brain" for recognizing numbers and set up a "waiter" (the API) to take orders and return results. We then put everything in a **Docker container**, which is like a standardized shipping crate that ensures the code runs the same way on any computer.

* **Script:** `train_model.py`
    * **Description:** Trains the baseline MNIST model and saves it as `mnist_model.h5`.
    * **Command:** `python3 train_model.py`
* **Script:** `app.py`
    * **Description:** Flask API that serves predictions using the trained model.
    * **Command:** `python3 app.py` (Server runs on port 5000).

**Docker Commands:**
* **Build:** `docker build -t mnist-app .`
* **Run:** `docker run -p 5000:5000 mnist-app`

---

### ✅ Practical 2: Improving Neural Networks
*Focus: Evolution of the training script with Regularization, MLOps, and Optimizers.*

This part is about **quality control**. We use "Regularization" (like Dropout) to act as a safety net, preventing the model from just memorizing the answers. We also use **MLflow**, which acts like a scientist's lab notebook to record every experiment automatically.



* **Script:** `train_model.py`
    * **Description:** Implements Data Splitting (Train/Val/Test), Bias/Variance diagnosis, Regularization (L2, Dropout, Batch Norm), and compares Optimizers using MLflow.
    * **Command:** `python3 train_model.py`

---

### ✅ Practical 3: CNNs & Computer Vision
*Focus: CNNs, ResNets, and Style Transfer.*

Now we give the machine "eyes." We use **CNNs**, which look at small patches of an image at a time (like a magnifying glass) to find patterns. We also explore **Style Transfer**, which is like teaching the computer to "paint" a photo in the style of a famous artist.

* **Script:** `cnn_classification.py`
    * **Description:** Trains a Basic CNN and implements a ResNet on CIFAR-10.
    * **Command:** `python3 cnn_classification.py`
* **Script:** `style_transfer.py`
    * **Description:** Uses VGG16 to extract Content and Style features from local images.
    * **Command:** `python3 style_transfer.py`

---

### ✅ Practical 4: Segmentation & 3D Data
*Focus: U-Net, Segmentation Metrics, and 3D Convolutions.*

Finally, we move to **high-precision work**. Instead of just identifying an object, we use a **U-Net** to outline it pixel by pixel (Segmentation)—similar to how a doctor carefully traces a tumor on a scan. We also look at 3D data, which is like looking at a whole stack of photos (a volume) instead of just one.



* **Script:** `unet_segmentation.py`
    * **Description:** Trains a U-Net on synthetic medical data using Dice and IoU metrics.
    * **Command:** `python3 unet_segmentation.py`
* **Script:** `conv3d_demo.py`
    * **Description:** Implements a 3D Convolutional block (Conv3D) and logs architecture to MLflow.
    * **Command:** `python3 conv3d_demo.py`

