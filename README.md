# 🎯 Digit Recognition using PyTorch

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.x-blue.svg?style=for-the-badge&logo=python)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)
![MNIST Dataset](https://img.shields.io/badge/Dataset-MNIST-orange?style=for-the-badge&logo=databricks)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

## 📚 Project Overview

This repository implements a **Convolutional Neural Network (CNN)** for handwritten digit recognition using the **MNIST** dataset (**we can also use custom dataset to train the model**), all within the PyTorch framework. The project leverages custom transformations to preprocess the dataset and provides utilities for saving, loading, and testing models, including a scripted version of the model for deployment.

## 📁 Repository Structure

The repository contains the following structure:

```bash
digit-recognition/
├── data/
│   └── MNIST/raw/        # MNIST dataset stored here
├── model/
│   ├── DigitRecognitionModel.pt          # Saved model
│   ├── DigitRecognitionModelParameter.pt # Model's state dictionary
├── Image recognition.ipynb   # Image classification experiments
├── digit_recognition.ipynb   # Main implementation of the project
├── scripted_model.pt         # Scripted model for testing and deployment
├── test.ipynb                # Scripted model testing
```

## 🚀 Key Features

- **Data Preprocessing**: The MNIST images are resized to 64x64 and converted to grayscale with 3 channels for consistency.
- **Model Architecture**: A CNN with two convolutional layers and fully connected layers, designed to classify digits (0-9).
- **Training**: The model is trained for 30 epochs multiple times with CrossEntropyLoss and Adam optimizer, tracking accuracy and loss on both training and test sets.
- **Model Saving/Loading**: The trained model and its parameters can be saved and loaded for further use.
- **Scripted Model**: A scripted version of the model is provided for easier deployment.

## 🛠️ Model Architecture

The model comprises the following layers:
- **Convolutional Layer 1**: `Conv2d(in_channels=3, out_channels=10, kernel_size=5)`
- **Max Pooling**: `MaxPool2d(kernel_size=2)`
- **ReLU Activation**: Activation function after each layer
- **Convolutional Layer 2**: `Conv2d(in_channels=10, out_channels=10, kernel_size=5)`
- **Dropout**: `Dropout2d()` to prevent overfitting
- **Fully Connected Layer**: Outputs predictions for the 10 possible digits

## ⚙️ Training Process

- **Loss Function**: `CrossEntropyLoss` to handle the multiclass classification.
- **Optimizer**: `Adam` optimizer with a learning rate of 0.01.
- **Accuracy Metric**: `torchmetrics.Accuracy` to evaluate performance during training and testing.
- **Epochs**: 30 epochs for multiple times.
- **Batch Size**: 50.

### Sample Output during Training:
```
Epoch : 0 | Train_loss : 0.137 | Train_acc : 96% | Test_loss : 0.148 | Test_acc : 96%
Epoch : 2 | Train_loss : 0.135 | Train_acc : 96% | Test_loss : 0.147 | Test_acc : 96%
...
Epoch : 28 | Train_loss : 0.115 | Train_acc : 97% | Test_loss : 0.134 | Test_acc : 96%
```

## 🔮 Prediction

The `MakePrediction` class allows you to predict digits from custom images by preprocessing them and feeding them into the trained model.

```python
a = MakePrediction('/path/to/image.jpg')
a.plot_img()
print(a.show_prediction().item())  # Outputs the predicted digit
```

## 💾 Model Saving and Loading

You can save the trained model and its parameters and load a previously saved model for future predictions.

```python
# Saving model
torch.save(model.state_dict(), 'model/DigitRecognitionModelParameter.pt')
torch.save(model, 'model/DigitRecognitionModel.pt')

# Loading model
loaded_model = torch.load('model/DigitRecognitionModel.pt')
loaded_model.eval()
```

## 📈 Future Improvements

- **Data Augmentation**: Implementing additional techniques like random rotation and scaling to improve the model’s ability to generalize.
- **Hyperparameter Tuning**: Experimenting with different learning rates, batch sizes, and optimizers to further enhance performance.
- **Deployment**: The scripted model can be deployed in real-time applications.

## 📋 Requirements

- Python 3.x
- PyTorch
- Torchvision
- Torchmetrics
- Pillow (PIL)
- Matplotlib
