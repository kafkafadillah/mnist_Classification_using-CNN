# mnist_Classification_using-CNN
This project demonstrates how to classify handwritten digits from the MNIST dataset using a Convolutional Neural Network (CNN). The model was trained using TensorFlow and TensorFlow Datasets (TFDS) for easy dataset loading and preprocessing.

# MNIST Digit Classification Using Convolutional Neural Network (CNN)

This project demonstrates how to classify handwritten digits from the MNIST dataset using a Convolutional Neural Network (CNN). The model was trained using TensorFlow and TensorFlow Datasets (TFDS) for easy dataset loading and preprocessing.

## Overview

The MNIST dataset contains 60,000 training images and 10,000 testing images of handwritten digits from 0 to 9. Each image is 28x28 pixels in grayscale.

In this project:
- The **MNIST dataset** is loaded using **TensorFlow Datasets (TFDS)**.
- A **Convolutional Neural Network (CNN)** is built and trained to classify these images into one of the 10 classes (digits 0-9).
- The model's performance is evaluated on the test set and its performance is measured using accuracy, loss, and a classification report.

## Requirements

- **TensorFlow 2.x**
- **TensorFlow Datasets (TFDS)**
- **Matplotlib**
- **Seaborn**
- **NumPy**

You can install the required dependencies using pip:

```bash
pip install tensorflow tensorflow-datasets matplotlib seaborn numpy
```

## Dataset

This project uses the MNIST dataset, which is a collection of handwritten digits. The dataset is available in TensorFlow Datasets (TFDS), which makes loading and preprocessing simple. The dataset consists of:
- 60,000 training images
- 10,000 testing images

### TensorFlow Datasets (TFDS)
- **Train Set**: Images and labels for training the model.
- **Test Set**: Images and labels used to evaluate the trained model.

## Model Architecture

The model architecture used is a Convolutional Neural Network (CNN) with the following layers:
1. **Conv2D**: 32 filters with a kernel size of 3x3 and ReLU activation function.
2. **MaxPooling2D**: Pooling layer with a 2x2 filter.
3. **Conv2D**: 64 filters with a kernel size of 3x3 and ReLU activation function.
4. **MaxPooling2D**: Pooling layer with a 2x2 filter.
5. **Conv2D**: 128 filters with a kernel size of 3x3 and ReLU activation function.
6. **MaxPooling2D**: Pooling layer with a 2x2 filter.
7. **Flatten**: Converts the 2D data into a 1D vector.
8. **Dense**: Fully connected layer with 64 neurons and ReLU activation.
9. **Dense**: Fully connected layer with 128 neurons and ReLU activation.
10. **Dense**: Output layer with 10 neurons (one for each class) and softmax activation.

## Training

The model is compiled using the **RMSprop** optimizer, **SparseCategoricalCrossentropy** loss function, and **accuracy** as the evaluation metric. The training process stops once the accuracy exceeds 96%, thanks to the custom callback function.

### Callbacks:
- **Early stopping**: The training halts early if the model's accuracy exceeds 96% on the training set.

## Results

- **Training Accuracy**: Achieved over 97% after two epochs.
- **Test Accuracy**: The model achieved **98.07%** accuracy on the test set.
- **Loss**: The test loss was **0.0686**.

## Confusion Matrix

![Confusion Matrix](./path/to/confusion_matrix_image.png)

The confusion matrix shows how well the model performs across the 10 digit classes. Most of the digits are correctly classified, with only a few errors on digits 8 and 5.

## Classification Report

The classification report includes precision, recall, and f1-score for each digit from 0 to 9, showing how well the model performs for each class.

```
              precision    recall  f1-score   support
           0       0.99      0.99      0.99       980
           1       0.99      1.00      0.99      1135
           2       0.95      0.99      0.97      1032
           3       0.99      0.98      0.99      1010
           4       0.99      0.96      0.98       982
           5       0.98      0.98      0.98       892
           6       0.99      0.98      0.99       958
           7       0.98      0.96      0.97      1028
           8       0.95      0.99      0.97       974
           9       0.98      0.97      0.98      1009

    accuracy                           0.98     10000
   macro avg       0.98      0.98      0.98     10000
weighted avg       0.98      0.98      0.98     10000
```

## Conclusion

The model achieves an excellent performance of **98.07%** accuracy on the test set, with a fast training time. The confusion matrix and classification report show that the model is highly effective at recognizing handwritten digits.

This project demonstrates how to use **TensorFlow** and **TensorFlow Datasets (TFDS)** to load and train a convolutional neural network for image classification.

## Future Improvements

- **Hyperparameter Tuning**: Experiment with different architectures, optimizers, and hyperparameters to improve model performance.
- **Data Augmentation**: Add more data augmentation techniques to increase the diversity of training data and reduce overfitting.
