# Training and testing dataset

A goal of supervised learning is to build a model that performs well on new data. Train test split is a model validation procedure that allows you to simulate how a model would perform on new/unseen data.

Here is how the procedure works:

![image](https://github.com/user-attachments/assets/cd54a948-d107-4314-a1b1-9e212d0cf43d)

The objective is to split the data set into two pieces: a training and a testing set.

This consists of random sampling without replacement of about 75 percent of the rows (you can vary this) and putting them into your training set. The remaining 25 percent is put into your test set.

Never train a model with test data. If you see surprisingly good results in your evaluation metrics, it could be a sign that you are accidentally training on the test suite. For example, a high precision may indicate that test data was leaked in the training set.

The scikit-learn library provides us with the model_selection module in which we have the splitter function train_test_split().

During this project, you will evaluate your skill to split the dataset into train and test sets.

Framingham Risk Score

The Framingham_10yrs dataset comprises a range of features that encompass the medical and lifestyle factors of individuals. The dataset provides valuable insights into the health profile of the subjects and can be utilized for diverse analytical purposes.

The dataset you will be using contains the following features:

![image](https://github.com/user-attachments/assets/f6b27c79-2ba9-4c7c-ab1a-1a178cc3586d)

# Notebook
![image](https://github.com/user-attachments/assets/86990899-6c2f-4f00-ab60-6e7ca963113a)

