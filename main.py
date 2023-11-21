import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy

# Load the dataset
dataset = pd.read_csv('/Users/tharunpeddisetty/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 3 - Classification/Section 14 - Logistic Regression/Python/Social_Network_Ads.csv')

# Separate features and target variable
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Splitting data into training and testing sets
np.random.seed(0)  # Set seed for reproducibility
indices = np.random.permutation(len(X))
split = int(0.75 * len(X))  # 75% training, 25% testing

X_train, X_test = X[indices[:split]], X[indices[split:]]
Y_train, Y_test = Y[indices[:split]], Y[indices[split:]]

# Feature Scaling
mean, std = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Build the neural network model
model = Sequential([
    Dense(units=6, activation='relu', input_dim=2),
    Dense(units=1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=[Accuracy()])

# Train the model
history = model.fit(X_train, Y_train, batch_size=32, epochs=50, verbose=1)

# Evaluate the model on the test set
accuracy = model.evaluate(X_test, Y_test, verbose=0)[1]
print("Test Accuracy:", accuracy)

# Plot the training history
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Predicting for age=30, estimated salary=87000
new_data = np.array([[30, 87000]])
new_data_scaled = (new_data - mean) / std
prediction = model.predict(new_data_scaled)
print("Predicted probability:", prediction)
