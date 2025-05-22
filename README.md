# CNN from Scratch in NumPy | MNIST Digit Classifier

This project implements a **Convolutional Neural Network (CNN)** entirely from scratch using **NumPy**—with no machine learning or deep learning libraries (like TensorFlow or PyTorch). The model is trained on the **MNIST** handwritten digits dataset.

---

## Project Structure

```
.
├── mnist.npz                   # MNIST dataset
├── notebook/code.ipynb            # All model code
└── README.md                   # This documentation
```

---

## Features

- ✅ CNN architecture built from scratch using only NumPy  
- ✅ Custom Layers:  
  - `Conv2d` – 2D convolution  
  - `ReLU` – Activation function  
  - `MaxPool2d` – Max pooling layer  
  - `BatchNormalization` – Normalization layer  
  - `Dropout` – Regularization  
  - `Flatten` – Prepare for dense layers  
  - `Linear` – Fully connected layers  
  - `SoftmaxWithCrossEntropyLoss` – Combined activation + loss  
- ✅ Manual implementation of forward and backward passes  
- ✅ Utility functions for:  
  - Efficient convolution via `im2col` and `col2im`  
  - Output size calculation  
  - Data loading and one-hot encoding  
- ✅ Custom `DataGenerator` for batching  
- ✅ End-to-end model training & evaluation  

---

## Dataset

- **MNIST** (Modified National Institute of Standards and Technology database)  
- 60,000 training images and 10,000 test images  
- 28x28 grayscale handwritten digit images (0–9)  
- Dataset file: `mnist.npz`  

---

## Getting Started

### 1. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Set Dataset Path

Update the path to where your `mnist.npz` is stored:

```python
path = "/content/drive/MyDrive/Colab Notebooks/DATA - 690 Deep Learning/mnist.npz"
```

### 3. Run the Notebook

- Upload `code.py` into Colab  
- Execute each cell sequentially  
- Training loop, accuracy tracking, and final testing will run  

---

## Model Architecture

```
Conv2d -> ReLU -> MaxPool2d -> BatchNormalization -> Dropout -> Flatten -> Linear -> ReLU -> Linear -> SoftmaxWithCrossEntropyLoss
```

### Sample Model Definition

```python
model = Sequential([
    Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
    ReLU(),
    MaxPool2d(kernel_size=2, stride=2),
    BatchNormalization(num_features=8),
    Dropout(p=0.25),
    Flatten(),
    Linear(in_features=8*14*14, out_features=128),
    ReLU(),
    Linear(in_features=128, out_features=10),
    SoftmaxWithCrossEntropyLoss()
])
```

---

## Evaluation

- Evaluated on 10,000 test images  
- Accuracy calculated manually  
- Print statements track loss/accuracy per epoch  

---

## Learning Outcomes

- Deep understanding of CNN internals  
- Confidence with backpropagation and forward passes  
- Practical understanding of convolution, pooling, dropout, normalization  
- Low-level implementation skills in NumPy  

---

## Future Improvements

- Add learning rate scheduler  
- Implement Adam optimizer  
- Confusion matrix & classification report  
- Visualizations for predictions and misclassifications  
- Save and load model weights  

---

## References

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)  
- Stanford CS231n Course Notes  
- NumPy Documentation  

---

## Author

**Drashi Dave**  
Graduate Student, DATA 690 – Deep Learning  
University of Maryland Baltimore County
