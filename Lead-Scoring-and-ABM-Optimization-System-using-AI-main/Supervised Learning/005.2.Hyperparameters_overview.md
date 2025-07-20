#### Model Parameters and Hyperparameters Overview

In machine learning, understanding the distinction between model parameters and hyperparameters is crucial. Model parameters are learned from the data during training, while hyperparameters are set before training and influence the learning process. In this lab, we'll explore these concepts using various machine learning models and their associated hyperparameters.

Example with Different Models and Hyperparameters
Let's explore the concept using examples of different machine learning models and their associated hyperparameters.

1- Support Vector Machine (SVM):
```
from sklearn.svm import SVC
model = SVC(C=1.0, kernel='linear')
2- Random Forest Classifier:
```
```
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=10)
```
3- Neural Network:
```
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(100, 50), alpha=0.001)
```
Practice

In this exercise, we will walk through the process of working with a dataset, choosing a machine learning model, training the model, evaluating its performance, and experimenting with hyperparameter values. This practical exercise will provide valuable insights into the interplay between hyperparameters and model performance.

Let's practice by yourself and then check your answer:

1- Load a dataset (e.g., Iris dataset) and split it into features (X) and target (y).

2- Choose a machine learning model from the examples above and create an instance of the model with different hyperparameters.

3- Train the model using the training data.

4- Evaluate the model's performance on the testing data.

5- Experiment with changing hyperparameter values and observe how they impact the model's performance.

Let's test your understanding with a multiple-choice activity. Choose the correct answer for each step of the exercise:

![image](https://github.com/user-attachments/assets/9ce95c28-f700-4be4-abe0-fbca4e12acf9)

![image](https://github.com/user-attachments/assets/97c4b5fd-039f-4a29-883e-7b1b62f0137c)

### QUIZ

![image](https://github.com/user-attachments/assets/7249f2c7-324d-421a-b955-2c2a462f02a9)
