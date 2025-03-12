import numpy as np
import matplotlib.pyplot as plt
import random
from models.house import House
from services.house_price_predictor import HousePricePredictor
from services.gradient_descent import GradientDescent
from services.feature_scaler import FeatureScaler
from services.mean_squared_error import Mse


#build training dataset
houses = []
for i in range(1,5):
    if i == 1:
        house = House(300 * i, 1.0 * i, i, 1 * i)
        houses.append(house)
    else:
        house = House((300 * i) - 100, 1.0 * i, random.randint(2,10), random.randint(1,20))
        houses.append(house)

for house in houses:
    print(f"House sqft: {house.sqtf}")
    print(f"House age: {house.age}")
    print(f"House rooms: {house.rooms}")
    print(f"House price: {house.price}")

y_train = np.array([house.price for house in houses])
m = len(houses)
features = []

for i in range(m):
    feature_i = [houses[i].sqtf, houses[i].rooms, houses[i].age]
    features.append(feature_i)

X_train = np.array(features)

#scale the data
m,n = X_train.shape
X_train_scaled = np.zeros((m,n))

for i in range(m):
    X_train_scaled[i] = FeatureScaler.simple_feature_scaling(X_train[i,:])
y_train_scaled = FeatureScaler.simple_feature_scaling(y_train)

predictor = HousePricePredictor()
mse = Mse()
w_in = None
b_in = 0
m,n = X_train_scaled.shape
w_in = np.zeros((n,))

print("First 5 Scaled Features:", X_train_scaled[:5])

#check gradient descent
w, b, history = GradientDescent.call(w_in, b_in, 0.0003, 2800, X_train_scaled, y_train_scaled, mse)
print(f"value of weight: {w}")
print(f"value of bias: {b}")

#plot cost function
plt.plot(history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs Iterations')
plt.show()


#plot pridection line
predictions = predictor.predict(w, b, X_train_scaled)



#check cost function
cost = mse.call(w, b, X_train_scaled, y_train_scaled)
print(f"cost of mse: {cost}")

print("X_train_scaled shape:", X_train_scaled.shape)  # Expected: (19, 3)
print("y_train_scaled shape:", y_train_scaled.shape)  # Expected: (19,)
print("Predictions shape:", predictions.shape)        # Expected: (19,)
print("Predictions:", predictions)                     # Check prediction values


# Get predictions for all scaled input data
predictions = predictor.predict(w, b, X_train_scaled)

# Create a 3D plot for visualization
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the training data as red 'x' markers
sc1 = ax.scatter(
    X_train_scaled[:, 0], 
    X_train_scaled[:, 1], 
    X_train_scaled[:, 2], 
    c=y_train_scaled, 
    cmap='viridis', 
    marker='x', 
    label='Training Data'
)

# Plot the predictions as blue points with a color map
sc2 = ax.scatter(
    X_train_scaled[:, 0], 
    X_train_scaled[:, 1], 
    X_train_scaled[:, 2], 
    c=predictions, 
    cmap='cool', 
    marker='o', 
    label='Predictions', 
    alpha=0.6
)

# Add text labels to points to show predicted price
for i in range(len(predictions)):
    ax.text(X_train_scaled[i, 0], X_train_scaled[i, 1], X_train_scaled[i, 2], 
            f"{predictions[i]:.2f}", fontsize=8, color='black')

# Set axis labels
ax.set_xlabel('House Sqft (Scaled)')
ax.set_ylabel('Number of Rooms (Scaled)')
ax.set_zlabel('Age of House (Scaled)')

# Add color bar to show the scale of predicted prices
cbar = plt.colorbar(sc2, pad=0.1)
cbar.set_label('Predicted Price (Scaled)')

plt.title('3D Plot of House Features vs Predicted Prices with Labels')
plt.legend()
plt.show()