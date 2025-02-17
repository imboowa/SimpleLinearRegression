from typing import Any

import numpy as np

# Prepare our data primitively
X = np.array(
    [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
     35, 36, 37, 38, 39, 40])
Y = np.array(
    [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76,
     78, 80, 82, 84, 86, 88, 90])

# data length
data_length = len(X)

# Prepare what is used in data scaling
X_mean = np.mean(X)
X_std = np.std(X)
Y_mean = np.mean(Y)
Y_std = np.std(Y)

# Scale our data
X_scaled = (X - X_mean) / X_std
Y_scaled = (Y - Y_mean) / Y_std

# Training dataset
x_train = X_scaled
y_train = Y_scaled

# variables for use
w, b, epochs, alpha = 0, 0, 10000, 1e-3

for _ in range(epochs):
    y_pred = (w * x_train) + b
    error = (y_train - y_pred)

    mse = (1 / data_length) * np.sum(error ** 2)

    dj_w = (-2 / data_length) * np.sum(error * x_train)
    dj_b = (-2 / data_length) * np.sum(error * 1)

    w = w - alpha * dj_w
    b = b - alpha * dj_b


def predict(age):
    scaled_age = (age - X_mean) / X_std
    scaled_prediction = (w * scaled_age) + b
    prediction = (scaled_prediction * Y_std) + Y_mean
    return np.round(prediction)


print(f"Weight : {w}\nBias : {b}\nMSE : {mse}\n")
print(f"Your Weight: {predict(5)}")
