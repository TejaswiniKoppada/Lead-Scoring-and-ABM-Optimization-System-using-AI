Hotel Reservations Dataset
The Hotel Reservations Dataset is a popular dataset for exploring and analyzing data related to hotel reservations. The dataset contains information about bookings made in two hotels: one resort hotel and one city hotel. The data includes information about the booking dates, the number of guests, the type of rooms reserved, and various other details related to the bookings.

In this lab, you will be using a hotel reservations dataset to validate your skills in machine learning. The lab is not guided, so you should follow the machine learning pipeline, including data cleaning, feature engineering, model training, etc., to achieve the best performance on a different dataset.

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the dataset
data = pd.read_csv('train.csv')

# Drop irrelevant columns
data = data.drop(columns=['Booking_ID', 'arrival_year', 'arrival_month', 'arrival_date'])

# Handle missing values (if any)
data = data.fillna(0)

# Encode categorical variables
categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Feature engineering
data['total_nights'] = data['no_of_weekend_nights'] + data['no_of_week_nights']
data['total_guests'] = data['no_of_adults'] + data['no_of_children']

# Separate features and target
X = data.drop(columns=['booking_status'])
y = data['booking_status']

# Encode target variable
y = y.map({'Not_Canceled': 0, 'Canceled': 1})

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
```
Performance

```
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, precision_score, recall_score
y_pred = model.predict(X_val)
print(f'Accuracy: {accuracy_score(y_val, y_pred)}')
print(f'Precision: {precision_score(y_val, y_pred)}')
print(f'Recall: {recall_score(y_val, y_pred)}')
```
Submission
```
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the training and test data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Save the Booking_ID column for submission
submission = pd.DataFrame()
submission['Booking_ID'] = test_data['Booking_ID']

# Drop irrelevant columns (same for both training and test data)
train_data = train_data.drop(columns=['Booking_ID', 'arrival_year', 'arrival_month', 'arrival_date'])
test_data = test_data.drop(columns=['Booking_ID', 'arrival_year', 'arrival_month', 'arrival_date'])

# Handle missing values (if any)
train_data = train_data.fillna(0)
test_data = test_data.fillna(0)

# Encode categorical variables using OneHotEncoder
categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
encoder = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)

# Fit the encoder on the training data and transform both training and test data
train_data_encoded = encoder.fit_transform(train_data[categorical_cols])
test_data_encoded = encoder.transform(test_data[categorical_cols])

# Convert encoded data to DataFrames
train_data_encoded = pd.DataFrame(train_data_encoded, columns=encoder.get_feature_names_out(categorical_cols))
test_data_encoded = pd.DataFrame(test_data_encoded, columns=encoder.get_feature_names_out(categorical_cols))

# Drop original categorical columns and concatenate encoded columns
train_data = train_data.drop(columns=categorical_cols)
test_data = test_data.drop(columns=categorical_cols)

train_data = pd.concat([train_data, train_data_encoded], axis=1)
test_data = pd.concat([test_data, test_data_encoded], axis=1)

# Feature engineering (same as training data)
train_data['total_nights'] = train_data['no_of_weekend_nights'] + train_data['no_of_week_nights']
test_data['total_nights'] = test_data['no_of_weekend_nights'] + test_data['no_of_week_nights']

train_data['total_guests'] = train_data['no_of_adults'] + train_data['no_of_children']
test_data['total_guests'] = test_data['no_of_adults'] + test_data['no_of_children']

# Separate features and target in the training data
X_train = train_data.drop(columns=['booking_status'])
y_train = train_data['booking_status']

# Encode target variable
y_train = y_train.map({'Not_Canceled': 0, 'Canceled': 1})

# Normalize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
test_data = scaler.transform(test_data)

# Train the model (example: RandomForestClassifier)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test data
test_predictions = model.predict(test_data)

# Map predictions back to 'Canceled' or 'Not_Canceled'
submission['booking_status'] = test_predictions
submission['booking_status'] = submission['booking_status'].map({0: 'Not_Canceled', 1: 'Canceled'})

# Save predictions to submission.csv
submission.to_csv('submission.csv', index=False)
```
