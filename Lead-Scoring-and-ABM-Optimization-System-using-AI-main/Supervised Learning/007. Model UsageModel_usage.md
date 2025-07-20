The objective of this lab is to introduce you to using a pre-trained machine learning model in Scikit-Learn for making predictions on new data. We will cover the steps involved in loading a pre-trained model from a file, uploading a new dataset, and using the model to make predictions on the new data.

Pre-Trained Model in Scikit-Learn

Load the Pre-Trained Model
First, we need to load the pre-trained model from a file. For this lab, we assume that you have already trained a model and saved it to a file using Scikit-Learn's model saving functionality. The saved model can be loaded back into memory when needed.

```
import joblib

# Load the pre-trained model from the file
model = joblib.load('pretrained_model.pkl')
```
Upload a New Dataset

Next, we'll upload a new dataset that we want to use for making predictions. The new dataset should have the same format as the data used to train the pre-trained model.
```
# Upload the new dataset for making predictions
new_data = np.array([[2.5], [3.0], [4.2], [5.1]])
```
Make Predictions

With the pre-trained model loaded and the new dataset uploaded, we can now use the model to make predictions on the new data.
```
# Make predictions on the new dataset
predictions = model.predict(new_data)
```

#### QUIZ

![image](https://github.com/user-attachments/assets/647baef6-a0ae-49de-b1d9-d69113d4a5e4)
