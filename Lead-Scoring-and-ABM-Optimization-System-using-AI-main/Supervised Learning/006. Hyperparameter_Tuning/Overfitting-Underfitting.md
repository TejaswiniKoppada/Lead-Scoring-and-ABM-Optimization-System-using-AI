#### Overfitting and Underfitting

In this session, we'll explore two critical concepts in machine learning: overfitting and underfitting. Understanding these concepts is essential for building models that generalize well to new data and make accurate predictions. 

In the realm of machine learning, achieving the right balance between model complexity and generalization is crucial. Two key phenomena that exemplify this balance are overfitting and underfitting.

Overfitting occurs when a model captures noise and fluctuations in the training data to an excessive degree, leading to poor performance on unseen data.

Underfitting arises when a model is too simplistic to capture the underlying patterns in the data, resulting in suboptimal predictions.

Understanding these concepts is fundamental for building models that can reliably generalize to new data, ensuring accurate predictions and actionable insights.

Example
The following example effectively illustrates the challenges of underfitting and overfitting, highlighting the application of linear regression with polynomial features to approximate non-linear functions. The plotted graph depicts the targeted function—an excerpt of the cosine function—along with actual samples from the function and the approximations from various models using different degrees of polynomial features.

![image](https://github.com/user-attachments/assets/d3ebc636-ec14-43ce-a28e-31e5ca202b1f)

It's apparent that a simple linear function (polynomial degree 1) struggles to capture the essence of the training samples, demonstrating underfitting. In contrast, a polynomial of degree 4 closely matches the true function, showcasing a balanced approximation. However, with higher polynomial degrees, the model overfits the training data by capturing noise.

To objectively assess the extent of overfitting and underfitting, we employ cross-validation and calculate the mean squared error (MSE) on the validation set. A higher MSE indicates a reduced likelihood of the model generalizing accurately from the training data. This evaluation metric provides a quantitative insight into the performance of the models and helps us determine the optimal degree of polynomial features for minimizing overfitting or underfitting.

##### Validation curve
A model that is underfit will have high training and high testing error while an overfit model will have extremely low training error but a high testing error. In this session, we'll delve into a fundamental concept of model evaluation in machine learning known as the validation curve. This graph nicely summarizes the problem of overfitting and underfitting.

![image](https://github.com/user-attachments/assets/71d21096-60a7-4b73-a0bb-588340f562b1)

In the above diagram, when the model complexity is low, the training and generalization error are both high. This represents the model underfitting. When the model complexity is very high, there is a very large gap between training and generalization error. This represents the case of model overfitting. The sweet spot is in between, represented using a dashed line. At the sweet spot, e.g., the ideal model, there is a smaller gap between training and generalization error.

In the context of machine learning, "complexity" refers to how intricate or intricate a model is in terms of its ability to capture relationships within data.

Practice validation curve
Here's an example code that demonstrates how to create and interpret a validation curve using Scikit-Learn.

In this example, we'll use a Support Vector Machine (SVM) classifier and vary the regularization parameter (C) to observe its impact on the model's performance.

We will delve deeper into the model in the next skill track.

First, load the Iris dataset ('iris_dataset.csv'), separate the target variable from the features, and create two separate variables for them.

The following code demonstrates the process of creating a validation curve for a Support Vector Machine (SVM) classifier. The param_range represents a range of values for the regularization parameter C.

The validation curve is constructed by calculating both training and cross-validation scores for each value of C. The mean and standard deviation of these scores are computed to understand the model's performance.

The resulting validation curve is plotted, showing the impact of C on the training and cross-validation scores.

Let's run the code in your notebook!

```
# Varying values of C
param_range = np.logspace(-3, 3, 10)

# Calculate validation scores for varying C values
train_scores, test_scores = validation_curve(
    SVC(), X, y, param_name="C", param_range=param_range, cv=5, scoring="accuracy"
)

# Calculate mean and standard deviation of scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot validation curve
plt.figure(figsize=(10, 6))
plt.title("Validation Curve with SVM")
plt.xlabel("C")
plt.ylabel("Score")
plt.semilogx(param_range, train_mean, label="Training score", color="blue")
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2, color="blue")
plt.semilogx(param_range, test_mean, label="Cross-validation score", color="orange")
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.2, color="orange")
plt.legend(loc="best")
plt.show()
```

#### Output
![image](https://github.com/user-attachments/assets/52569eaf-d0f4-4714-96d8-784cb2c93a12)

### QUIZ
![image](https://github.com/user-attachments/assets/31e6e581-dd8d-4a23-b4fa-f2e27e65cbc1)
