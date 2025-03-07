import numpy as np
import matplotlib.pyplot as plt
from models.house import House
from services.house_price_predictor import HousePricePredictor
from services.gradient_descent import GradientDescent
from services.feature_scaler import FeatureScaler
from services.mean_squared_error import Mse


#build training dataset
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

#scale the data
x_train_scaled = FeatureScaler.simple_feature_scaling(x_train)
y_train_scaled = FeatureScaler.simple_feature_scaling(y_train)

predictor = HousePricePredictor()
mse = Mse()
#check gradient descent
w, b, history = GradientDescent.call(0, 0, 0.1, 1000, x_train_scaled, y_train_scaled, mse, predictor)
print(f"value of weight: {w}")
print(f"value of bias: {b}")

#plot cost function
plt.plot(history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs Iterations')
plt.show()


#plot pridection line
predictions = predictor.predict(w, b, x_train_scaled)


#check cost function
cost = mse.call(predictions, x_train_scaled, y_train_scaled)
print(f"cost of mse: {cost}")


#plot prediction limne
plt.plot(x_train_scaled, predictions, c='blue')

#build a scatter plot
plt.scatter(x_train_scaled, y_train_scaled, c='red', marker='x')
plt.xlabel('House Sqfts')
plt.ylabel('House Prices')
plt.title('House Prices vs Sqfts')
plt.show()