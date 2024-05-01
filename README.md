## Dog-Vs-Cat-Classification using MobileNet

# Overview
This project implements a deep learning model to classify images of cats and dogs using the MobileNet architecture as a pre-trained model. The model is fine-tuned on a dataset consisting of labeled images of cats and dogs to achieve high accuracy in distinguishing between the two classes.

# Requirements
Python 3.x
TensorFlow 2.x
Keras
NumPy
Matplotlib
Jupyter Notebook (optional, for running the provided notebooks)

# Dataset
The dataset used for training and evaluation consists of a collection of images of cats and dogs from [Kaggle mobilenet_v2](https://www.kaggle.com/models/google/mobilenet-v2/frameworks/tensorFlow2/variations/tf2-preview-feature-vector/versions/4?tfhub-redirect=true) . Each image is labeled as either a cat or a dog, and the dataset is split into training and validation sets for model training and evaluation.

# Model Architecture
The model architecture is based on the MobileNet architecture, which is a lightweight and efficient convolutional neural network suitable for mobile and embedded devices. The MobileNet model is pre-trained on the ImageNet dataset and fine-tuned on the cat and dog dataset to adapt it to the specific classification task.

