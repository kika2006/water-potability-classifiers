# Is This Water Safe to Drink? Naive Bayes and a Neural Network from Scratch

**Author:** Francisca (Kika) Freixo  

Predicting water potability from 9 chemical measurements with two classifiers written **from scratch in base R**, without ML libraries: a **Gaussian Naive Bayes** classifier and a **multi-layer perceptron** trained with backpropagation.

**R · probability · gradient descent · model evaluation**

---

## What's inside
- **Preprocessing**: train/test split and mean imputation using training-set means only, to avoid leakage. Features are z-normalised for the neural network.
- **Gaussian Naive Bayes**: class-conditional Gaussian parameters for each feature, computed with log densities for numerical stability. Fitted densities are plotted for features such as pH and chloramines.
- **Multi-layer perceptron**: weight initialisation, ReLU hidden layers, a sigmoid output, cross-entropy loss and 4,000 epochs of gradient descent, implemented by hand. Includes the loss curve and a visualisation of the hidden-neuron weights.
- **Evaluation**: confusion matrices, false-positive and false-negative rates (the costly mistake here is calling unsafe water safe), test accuracy of both models, and predictions for a new measurement.

## Running it
```r
# put water.csv in data/ (3,276 daily measurements + potability label)
source("potability_classifiers.R")
```

*Maastricht University, Machine Learning (2025), group assignment. The dataset was provided by the course and is similar to the public [Kaggle Water Potability dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability).*
