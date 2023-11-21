import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def load_data(file_path):
    # Load dataset
    return pd.read_csv(file_path)

def preprocess_data(dataset):
    # Extract features and target variable
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1].values
    
    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

    # Feature Scaling (Standardization)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test, Y_train, Y_test

def build_neural_network(input_dim):
    # Build the neural network
    model = Sequential()
    model.add(Dense(units=6, activation='relu', input_dim=input_dim))
    model.add(Dense(units=6, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def train_neural_network(model, X_train, Y_train, epochs=100, batch_size=32):
    # Train the model
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

def make_prediction(model, new_data, scaler):
    # Make a prediction for new data
    scaled_data = scaler.transform(np.array([new_data]))
    return model.predict(scaled_data)[0][0]

def evaluate_model(model, X_test, Y_test):
    # Predicting for test data
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

    # Display predictions alongside actual values
    print(np.concatenate((y_pred.reshape(len(y_pred), 1), Y_test.reshape(len(y_pred), 1)), 1))

    # Confusion Matrix
    cm = confusion_matrix(Y_test, y_pred)
    print(cm)
    
    # Accuracy Score
    print(f'Accuracy Score: {accuracy_score(Y_test, y_pred)}')

def main():
    # Load and preprocess data
    dataset = load_data('/Users/tharunpeddisetty/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 3 - Classification/Section 14 - Logistic Regression/Python/Social_Network_Ads.csv')
    X_train, X_test, Y_train, Y_test = preprocess_data(dataset)

    # Build neural network
    input_dim = X_train.shape[1]
    model = build_neural_network(input_dim)

    # Train neural network
    train_neural_network(model, X_train, Y_train)

    # Make prediction for new data
    new_data = [30, 87000]
    prediction = make_prediction(model, new_data, StandardScaler())

    print(f'Prediction for new data: {prediction}')

    # Evaluate the model
    evaluate_model(model, X_test, Y_test)

if __name__ == "__main__":
    main()
