# LightGTS – EfficientNet-B0 Image Classification Project

This project implements a LightGTS-based convolutional neural network (CNN) using EfficientNet-B0 as its feature
extraction backbone, built with PyTorch.
The model classifies playing card images (e.g., five of diamonds, king of hearts, etc.) into their respective categories
with high accuracy and computational efficiency.

---

## Model Overview

LightGTS (Light Gradient Transition Structure)

LightGTS is a lightweight deep learning architecture that emphasizes:

- Low computational cost

- High feature transferability

- Improved gradient flow through skip and transition layers

- Compactness for deployment on limited hardware

It’s designed to be modular — meaning the backbone network can be replaced or combined with other architectures.

--- 

## EfficientNet-B0 as Backbone

In this project, EfficientNet-B0 serves as the core feature extractor inside the LightGTS framework.

Why EfficientNet-B0?

- Uses compound scaling — balancing depth, width, and resolution.

- Achieves state-of-the-art accuracy-to-parameter ratio.

- Extremely efficient on GPU and CPU.

- Pretrained weights available via timm or torchvision.

By embedding EfficientNet-B0 inside LightGTS, we combine:

- EfficientNet’s optimized convolution blocks (MBConv)

- LightGTS’s transition layers for faster inference and smoother gradient propagation.

--- 

## Requirements

You will need the following libraries to run this project:

- Python 3.x
- NumPy
- Matplotlib
- pandas
- tqdm
- pillow
- timm
- torch
- torchvision

### Install requirements

Make sure Python 3.x is installed.

You can install the required packages by running:

1. Clone the repository:
   ```bash
   git clone https://github.com/developsa/cardClassifier.git
   ```
2. Navigate into the project directory:
   ```bash
   cd card_classifier
   ```
3. Install the necessary libraries:
    ```bash
    pip install -r requirements.txt
   ```
4. Run the project:
    ```bash
   python src/test.py
   ```

### Final Results

| **Metric**         | **Description**         | **Result**  |
|:-------------------|:------------------------|:------------|
| **Accuracy**       | Classification accuracy | **94.8%**   |
| **Loss**           | Cross-entropy loss      | **0.19**    |
| **Inference Time** | Per image (on GPU)      | **< 45 ms** |


## PREDICTION

![cardPrediction.png](cardPrediction/cardPrediction.png)
![cardPrediction.png2](cardPrediction/cardPrediction2.png)
![cardPrediction.png3](cardPrediction/cardPrediction3.png)


