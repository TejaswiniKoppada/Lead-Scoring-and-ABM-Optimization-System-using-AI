#### Creating Decision Tree from scratch.   
In this lab, we are going to program the query of a decision tree and then calculate some Gini impurities. Scikit-Learn automatically implements all of this, but doing it yourself will help you better understand decision trees.

It is a simple lab with the main objective of introducing you to the fundamentals of one of the simplest models.

#### Gini index
The Gini index measures the probability of misclassifying a randomly chosen element from the set, and it ranges from 0 (perfectly pure) to 1 (maximally impure). A split with a lower Gini index is considered better than a split with a higher Gini index.

The Gini index is calculated as follows:


where 
 is the proportion of examples in the 
 class in the set.

The minimum value of the Gini Index is 0. This happens when the node is pure, this means that all the contained elements in the node are of one unique class. Therefore, this node will not be split again. Thus, the optimum split is chosen by the features with less Gini Index. Moreover, it gets the maximum value when the probability of the two classes are the same.

Suppose we have a dataset of animals, where each animal is described by its features: whether it has fur, whether it lays eggs, and whether it is a mammal or a reptile. We want to build a decision tree to classify animals as either mammals or reptiles based on these features.

We can start by computing the Gini index for the original dataset, which has 10 animals, 6 of which are mammals and 4 of which are reptiles. The Gini index for this set is:
```
# calculate the Gini index
gini_i= 1 - (6/10)**2- (4/10)**2 #=0.48

print("Gini Index:", gini_i)
```
Next, we can split the dataset based on whether the animal has fur or not. If an animal has fur, we can assume that it is more likely to be a mammal than a reptile.

Suppose that when we split the dataset based on whether the animal has fur, we end up with two subsets:

one with 7 animals (5 mammals and 2 reptiles) that have fur,
and another with 3 animals (1 mammal and 2 reptiles) that do not have fur.
The Gini index for each of these subsets is:
```
# calculate the Gini index
gini_fur= 1 - (5/7)**2- (2/7)**2 #=0.408
gini_nofur= 1 - (1/3)**2- (2/3)**2 #=0.444
print("Gini Index fur:", gini_fur)
print("Gini Index nofur:", gini_nofur)
We can then calculate the weighted Gini index:
```
```
# calculate the Gini index
Gini_w= 7/10*gini_fur+3/10*gini_nofur #=0.4196
print("Gini Index weighted:", Gini_w)
print("Gini Index nofur:", gini_nofur)
```
If we compare this value to the Gini index of the original dataset (0.48), we see that the split based on fur results in a lower Gini index, indicating that it is a better split than not considering the fur feature.

This is just a simple example, and in practice, decision trees can be more complex and involve many more features and splits. The Gini index can be used to evaluate the quality of each split and determine the best path through the tree.

##### Conclusion
In conclusion, this lab provided an introduction to the fundamentals of decision tree models, including the Gini index and the CART algorithm. Through the analysis of example datasets, we gained insight into how decision trees can be used for classification tasks and how they can be interpreted to understand the factors that contribute to predictions. While this lab only scratched the surface of decision tree modeling, it provides a solid foundation for further exploration into more complex models and datasets.


#### QUIZ

![image](https://github.com/user-attachments/assets/67cfa6e6-4fa9-4c00-8006-021c51253bae)
![image](https://github.com/user-attachments/assets/3003b9f1-3da6-4a8d-8a48-f202e022ab44)
![Uploading image.png…]()
