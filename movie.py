import pandas as pd
import numpy as np

df = pd.read_csv('movie_test_data.csv')

X = df[['action_level', 'romance_level']].values
y = df['like_movie'].values

print("X SHAPE:")
print(X.shape)

print("Y SHAPE:")
print(y.shape)