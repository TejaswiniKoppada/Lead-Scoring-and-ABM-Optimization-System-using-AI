#### Classification algorithms-Decision Tree
In this lab, we start by working on Classification problems, one of the most important tasks within Machine Learning. Classification is a subcategory of supervised learning where the goal is to predict the categorical class label of new instances based on past observations. We will learn about the binary classification, meaning that N=2 (where N is a number of classes).

Some examples include:

An online banking service must be able to determine whether or not a transaction performed on the site is fraudulent.

Identifying if an email is a spam or not.

Transforming descriptions of medical diagnoses or procedures into standardized statistical code in a process known as clinical coding.

Identifying if an image is a cat or a dog.

##### Binary Classification
Generate Your Sample Dataset
We will automatically generate a group of 1000 instances with features called 
 and 
 - grouped in a single variable X- to which we assign a label y, which can be 0 and 1. We will do this by using a Scikit-Learn function:make_blobs.

Let's run the code in the notebook.
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=1000, centers=2,
                  random_state=0, cluster_std=1.3)
```
Now, we will make a figure to show the different instances that we generate as points in the (x1,x2) plane and assign them a different color depending on their y label:

Let's run the code in this notebook.
```
plt.scatter(X[:, 0], X[:, 1], c=y, s=25, cmap='bwr')
plt.colorbar()
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
```
![image](https://github.com/user-attachments/assets/4dd0b5c4-761c-4e50-9f32-bc527ba020e1)

##### Simple classification algorithm
Based on previous figure, we can define a simple classification:

If X2 is higher than 2 the new point corresponded to the blue group.

If X2 is lower than 2 the point corresponded to the red group.

![image](https://github.com/user-attachments/assets/cad52151-3dc4-47cf-a9b8-23d85552e3d3)

![image](https://github.com/user-attachments/assets/d11d30f3-bbc7-40e1-86ac-cf25f675e2c3)

We just developed a simple decision tree algorithm:

***Decision Trees are a type of Supervised Machine Learning where the data is continuously split according to a certain parameter. The tree can be explained by two entities, namely decision nodes, and leaves. The leaves are the decisions or the outcomes, and the decision nodes are where the data is split. Each split is based on a specific feature and threshold, and it determines the class or outcome associated with specific range of feature values.***

Let's visualize the decision rules in the scatterplot. A decision rule is a set of conditions or criteria that a classifier uses to assign class labels to instances in a dataset. The decision rule defines how the model makes predictions based on the features of the data.

#### QUIZ CLASSIFICATION
![image](https://github.com/user-attachments/assets/94b11e56-9285-4b15-8085-f04c6a73149b)
