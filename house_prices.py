import numpy as np
import matplotlib.pyplot as plt
from models.house import House

#build predicting model
def f_wb(w, b, x):
    #amount of data points
    n = x.shape[0]
    predictions = np.zeros(n)

    for i in range(n):
        predictions[i] = w * x[i] + b
    return predictions

#cost function
#formula > J(w,b) = 1/2m * sum (predictions[i] - expected result for i)^2
def mse(predictions, x, y):
    m = x.shape[0]
    sum_of_predictions = 0

    for i in range(m):
        sum_of_predictions += (predictions[i] - y[i]) ** 2

    return 1/(2*m) * sum_of_predictions

houses = []
for i in range(1,6):
    if i == 1:
        house = House(300 * i, 1.0 * i)
        houses.append(house)
    else:
        house = House((300 * i) - 100, 1.0 * i)
        houses.append(house)

    houses.append(house)

y_train = np.array([house.price for house in houses])
x_train = np.array([house.sqtf for house in houses])


#plot pridection line
w = 270
b = 10
predictions = f_wb(w, b, x_train)


#check cost function
cost = mse(predictions, x_train, y_train)
print(cost)


#plot prediction limne
plt.plot(x_train, predictions, c='blue')

#build a scatter plot
plt.scatter(x_train, y_train, c='red', marker='x')
plt.xlabel('House Sqfts')
plt.ylabel('House Prices')
plt.title('House Prices vs Sqfts')
plt.show()