# 🧠 PyTorch Modular CNN Pipeline

A modular deep learning project for **image classification using PyTorch and Convolutional Neural Networks (CNNs)**.

This project transforms a Jupyter Notebook-based image classification workflow into a structured Python application developed with **PyCharm**. It separates data preprocessing, CNN architecture, model training, evaluation, checkpoint saving, and inference into reusable modules.

## 🚀 Project Overview

The main goal of this project is to build a clean, reusable, and maintainable image classification pipeline using PyTorch.

Instead of implementing the entire deep learning workflow inside a single Jupyter Notebook, the project adopts a modular architecture where each Python file has a specific responsibility.

### ✨ Key Features

- **Custom CNN Architecture** — Image classification using a convolutional neural network built with PyTorch.
- **Modular Project Structure** — Independent modules for data loading, model creation, training, evaluation, and inference.
- **Image Preprocessing** — Resizing, tensor conversion, and channel-wise normalization.
- **Training & Evaluation** — Tracking training loss, training accuracy, test loss, and test accuracy.
- **Model Checkpointing** — Saving trained model parameters as `.pth` files.
- **Image Inference** — Loading a saved model and making predictions on new images.
- **Reusable Components** — Organized Python functions for easier maintenance and experimentation.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning framework |
| Torchvision | Image transformations and data loading |
| CNN | Image feature extraction and classification |
| PyCharm | Development environment |
| Git & GitHub | Version control and project management |

## 📁 Project Structure

```text
PyTorch-Modular-CNN-Pipeline/
│
├── data/
│   ├── desert101/
│   │   ├── train/
│   │   └── test/
│   └── baklava-online.jpg
│
├── models/
│   └── desert_classifier.pth
│
├── main.py
├── model_creation.py
├── setup_data.py
├── training_testing_engine.py
├── load_model_make_prediction.py
├── utils.py
├── requirements.txt
├── .gitignore
└── README.md
```

### Module Descriptions

| File | Description |
|---|---|
| `main.py` | Entry point for the training pipeline |
| `model_creation.py` | Defines the custom CNN architecture |
| `setup_data.py` | Creates training and testing DataLoaders |
| `training_testing_engine.py` | Implements training and evaluation loops |
| `load_model_make_prediction.py` | Loads trained weights and performs inference |
| `utils.py` | Provides utilities for saving models |
| `requirements.txt` | Contains project dependencies |

## 🔄 Machine Learning Pipeline

The project follows a structured deep learning workflow:

```text
Image Dataset
      ↓
Data Loading
      ↓
Image Preprocessing
      ↓
Custom CNN Model
      ↓
Training
      ↓
Evaluation
      ↓
Save Model Weights
      ↓
Load Trained Model
      ↓
Predict New Images
```

## 🖼️ Data Preprocessing

Images are resized to **64 × 64 pixels** and converted into PyTorch tensors.

Channel-wise normalization is applied using the dataset's calculated mean and standard deviation values.

```python
data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5483, 0.4638, 0.3865],
        std=[0.2173, 0.2279, 0.2263]
    )
])
```

Normalization helps standardize the input values used during model training.

The same normalization parameters are also applied during inference to maintain consistency.

## 🧠 Model Architecture

The project uses a custom Convolutional Neural Network implemented with PyTorch.

The network accepts RGB images with three input channels and produces classification scores corresponding to the dataset's classes.

The architecture is defined in `model_creation.py`, allowing the model to be reused independently from the training pipeline.

## ⚙️ Training Configuration

The training pipeline uses the following hyperparameters:

| Parameter | Value |
|---|---|
| Epochs | 10 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Hidden Units | 32 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Input Image Size | 64 × 64 |

During training, the pipeline tracks:

- Training Loss
- Training Accuracy
- Test Loss
- Test Accuracy

These metrics help monitor learning progress and evaluate classification performance.

## 💾 Model Saving and Loading

Once training is complete, the model parameters are saved using PyTorch.

```python
torch.save(model.state_dict(), "models/desert_classifier.pth")
```

Saved model parameters can later be loaded without retraining:

```python
model.load_state_dict(
    torch.load(
        "models/desert_classifier.pth",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()
```

This enables the trained network to be reused for inference on new images.

## 🔍 Making Predictions

The inference pipeline loads a trained CNN, preprocesses an image, and predicts its class.

```python
model.eval()

with torch.inference_mode():
    logits = model(single_image)
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = probabilities.argmax(dim=1).item()
```

The image must be preprocessed and provided as a batch tensor with shape:

```text
[Batch Size, Channels, Height, Width]
[1, 3, 64, 64]
```

The predicted index is mapped to the corresponding class name.

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/onursvm/PyTorch-Modular-CNN-Pipeline.git

cd PyTorch-Modular-CNN-Pipeline
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset

Organize your dataset into separate training and testing folders:

```text
data/
└── desert101/
    ├── train/
    │   ├── class_1/
    │   ├── class_2/
    │   └── ...
    └── test/
        ├── class_1/
        ├── class_2/
        └── ...
```

Each class should have its own directory containing the corresponding images.

The dataset is not included in the repository and must be provided separately.

### 5. Train the Model

```bash
python main.py
```

The script initializes the CNN, loads the dataset, trains the model, evaluates its performance, and saves the trained weights.

### 6. Run Inference

After training, place the image to be classified at the path configured in `load_model_make_prediction.py` and run:

```bash
python load_model_make_prediction.py
```

The script loads the saved model and prints the predicted class.

## 📓 From Notebook to Modular Python

This project is a modular implementation of my earlier Jupyter Notebook-based CNN classification work.

The original project focuses on exploring and understanding the deep learning workflow interactively.

This version focuses on organizing that workflow into reusable Python modules with separate responsibilities.

**Original Notebook Project:**

[PyTorch-CNN-Image-Classification](https://github.com/onursvm/PyTorch-CNN-Image-Classification)

### What Changed?

| Notebook Version | Modular Version |
|---|---|
| Interactive Jupyter Notebook | Standalone Python scripts |
| Experiment-oriented workflow | Organized training pipeline |
| Code in notebook cells | Separate reusable modules |
| Interactive model experiments | Dedicated training and inference entry points |
| Notebook-based execution | Command-line execution |

## 🎯 Learning Outcomes

Through this project, I practiced:

- Developing custom CNN models using PyTorch.
- Understanding image tensors and batch dimensions.
- Implementing preprocessing and normalization pipelines.
- Working with DataLoaders and mini-batch training.
- Applying forward propagation and backpropagation.
- Evaluating classification models using loss and accuracy.
- Saving and restoring model parameters.
- Performing inference on previously unseen images.
- Refactoring notebook experiments into modular Python applications.


## 👨‍💻 Author

**Onur Sevim**

GitHub: [@onursvm](https://github.com/onursvm)

---

⭐ If you find this project useful, consider giving the repository a star!
