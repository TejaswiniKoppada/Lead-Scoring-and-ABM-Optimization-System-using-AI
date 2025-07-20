In the context of machine learning models, there is a distinction between model parameters and hyperparameters. Here's an explanation of the difference:

#### Model Parameters:

Model parameters are the internal variables or weights that are learned by the model during the training process.
These parameters directly affect the predictions made by the model.
The values of model parameters are estimated by the model itself based on the training data.
Examples of model parameters include the coefficients in linear regression, the weights in neural networks, or the support vectors in support vector machines.

#### Hyperparameters:

Hyperparameters are external settings or configurations that are set before the training process begins.
These parameters control the behavior of the learning algorithm and influence how the model is trained.
Hyperparameters are not learned from the data, but rather set by the user or determined through techniques like cross-validation.
Examples of hyperparameters include the learning rate in gradient descent, the number of hidden layers in a neural network, or the regularization strength in a regularization algorithm.
In summary, model parameters are learned by the model during training and directly impact the predictions, while hyperparameters are set by the user and influence how the model is trained but do not directly affect the predictions.

For example, suppose you are building a support vector machines (SVM) classifier in sklearn:

```
from sklearn import svm
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC(C =0.01, kernel ='rbf', random_state=33)
clf.fit(X, y)
``` 
In the above code an instance of SVM is your estimator for your model for which the hyperparameters, in this case, are C and kernel. But your model has another parameter which is not a hyperparameter and that is random_state.

### QUIZ

![image](https://github.com/user-attachments/assets/0284ba65-1286-4097-b91a-ba9c59ec3040)
