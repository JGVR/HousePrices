import numpy as np
import matplotlib.pyplot as plt
from models.house import House

#build predicting model
def f_wb(w, b, x):
    #amount of data points
    m = x.shape[0]
    predictions = np.zeros(m)

    for i in range(m):
        predictions[i] = w * x[i] + b
    return predictions

# > gradient of the cost function 
# > aJ(w,b)/aw, aj(w,b)/ab
def gradient(w, b, x, y):
    d_w = 0
    d_b = 0
    m = x.shape[0]

    for i in range(m):
        f_wb = w * x[i] + b
        d_w += (f_wb - y[i]) * x[i]
        d_b += (f_wb - y[i])

    d_w = d_w / m
    d_b = d_b / m
    return d_w, d_b

#cost function
#formula > J(w,b) = 1/2m * sum (predictions[i] - expected result for i)^2
def mse(predictions, x, y):
    m = x.shape[0]
    sum_of_predictions = 0

    for i in range(m):
        sum_of_predictions += (predictions[i] - y[i]) ** 2

    return 1/(2*m) * sum_of_predictions

#gradient descent
# > w = w - alpha * dJ(w,b)/dw
# > b = b - alpha * dJ(w,b)/db
def gradient_descent(w_init, b_init, alpha, iters, x, y):
    w = w_init
    b = b_init
    history = []

    for i in range(iters):
        d_w, d_b = gradient(w, b, x, y)
        w = w - alpha * d_w
        b = b - alpha * d_b

        #store the cost per prediction
        predictions = f_wb(w, b, x)
        cost = mse(predictions, x, y)
        history.append(cost)

    return w, b, history

houses = []
for i in range(1,20):
    if i == 1:
        house = House(300 * i, 1.0 * i)
        houses.append(house)
    else:
        house = House((300 * i) - 100, 1.0 * i)
        houses.append(house)

    houses.append(house)

y_train = np.array([house.price for house in houses])
x_train = np.array([house.sqtf for house in houses])

#check gradient descent
w, b, history = gradient_descent(0, 0, 0.001, 100000, x_train, y_train)
print(f"value of weight: {w}")
print(f"value of bias: {b}")

#plot cost function
plt.plot(history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs Iterations')
plt.show()


#plot pridection line
predictions = f_wb(w, b, x_train)


#check cost function
cost = mse(predictions, x_train, y_train)
print(f"cost of mse: {cost}")


#plot prediction limne
plt.plot(x_train, predictions, c='blue')

#build a scatter plot
plt.scatter(x_train, y_train, c='red', marker='x')
plt.xlabel('House Sqfts')
plt.ylabel('House Prices')
plt.title('House Prices vs Sqfts')
plt.show()