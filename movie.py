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

