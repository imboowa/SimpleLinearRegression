import numpy as np
import random as rd

# This is the small training set
data_square_feet = np.array([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000,
                             6500, 7000])
data_rooms = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
data_price = np.array([10000, 50000, 100000, 150000, 200000, 250000, 250000, 300000, 350000, 400000, 450000,
                       500000,550000,600000])

# uniformly randomize weights for first try
w1, w2, b, alpha, epochs = rd.uniform(0,1), rd.uniform(0,1), rd.uniform(0,1), 1e-8, 10000

# Iterate for number of epochs tweeking weights and biases
for _ in range(epochs):
    y_pred = (w1 * data_rooms) + (w2 * data_square_feet) + b
    error = y_pred - data_price

    w1 = w1 - alpha * (2 / len(data_price) * np.sum(error * data_rooms))
    w2 = w2 - alpha * (2 / len(data_price) * np.sum(error * data_square_feet))

    b = b - alpha * (2 / len(data_price) * np.sum(error))

# After the iterations, provide the final weights and biases
print('Final Trained Weights')
print('Weight 1:  ', w1)
print('Weight 2:  ', w2)
print('\nBias:   ', b)
print()

# Try out new number of rooms and square feet
def predict(rooms, square_feet):
    return (w1 * rooms) + (w2 * square_feet) + b

# Print the predition
print('Predicted Price:    ',f'${predict(2,500):,.2f}')

