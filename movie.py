import pandas as pd
import numpy as np

df = pd.read_csv('movie_test_data.csv')

X = df[['action_level', 'romance_level']].values
y = df['like_movie'].values

print("X SHAPE:")
print(X.shape)

print("Y SHAPE:")
print(y.shape)

weights = np.zeros(2)
bias = 0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_probability(X, weights, bias):
    z = np.dot(X, weights) + bias
    return sigmoid(z)

def calculate_loss(y, y_pred):
    loss = -np.mean(y * np.log(y_pred) + (1-y) * np.log(1-y_pred))
    return loss

def calculate_gradients(X, y, y_pred):
    m = len(y)
    dw = np.dot(X.T, (y_pred - y)) / m
    db = np.sum(y_pred - y) / m
    return dw, db


learning_rate = 0.01
epochs = 10000

for epoch in range(epochs):

    y_pred = predict_probability(X, weights, bias)

    loss = calculate_loss(y, y_pred)

    dw, db = calculate_gradients(X, y, y_pred)

    weights = weights - learning_rate * dw
    bias = bias - learning_rate * db

print(weights)
print(bias)


# calculate the accuracy of the model
def Metrix(y_pred,y):
    TP = np.sum((y == 1) & (y_pred >= 0.5))
    TN = np.sum((y == 0) & (y_pred < 0.5))
    FP = np.sum((y == 0) & (y_pred >= 0.5))
    FN = np.sum((y == 1) & (y_pred < 0.5))
    print(TP, TN, FP, FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print("Accuracy:", accuracy)
    precision = TP / (TP + FP) 
    print("Precision:", precision)
    recall = TP / (TP + FN)
    print("Recall:", recall)

print(Metrix(y_pred, y))

#NOW TESTING THE MODEL
test_df = pd.read_csv('movie_test_data.csv')
X_test = test_df[['action_level', 'romance_level']].values
y_test = test_df['like_movie'].values

def predict_class(X, weights, bias):
    yT_pred = predict_probability(X_test, weights, bias)
    return (yT_pred >= 0.5).astype(int)

y_TEST_pred_class=predict_class(X_test, weights, bias)
print("FOR TEST DATA")
print(Metrix(y_TEST_pred_class, y_test))