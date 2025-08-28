# import os
# import numpy as np
# import joblib as jb
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score, mean_squared_error

# pizza_data = pd.read_csv('pizza_delivery_data.csv')
# features = ['distance_miles','pizza_count','day_of_week','weather','traffic_level']
# X = pizza_data[features]
# Y = pizza_data['delivery_time']

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# print(f'\nData Training on {len(X_train)} deliveries')
# print(f'Data Testing on {len(X_test)} deliveries')


# # Clean the target variable
# Y_train_clean = pd.to_numeric(Y_train, errors='coerce')
# nan_count = Y_train_clean.isna().sum()
# print(f'Found {nan_count} NaN values in Y_train')

# # Remove rows with NaN values from both X_train and Y_train_clean
# if nan_count > 0:
#     # Get indices where Y_train_clean is not NaN


#     valid_indices = ~Y_train_clean.isna()   # '~' is used as a not operator in pandas

#     # Filter both X and Y to remove NaN rows
#     X_train_final = X_train[valid_indices]
#     Y_train_final = Y_train_clean[valid_indices]

#     print(f'After cleaning: Training on {len(X_train_final)} deliveries')
# else:
#     X_train_final = X_train
#     Y_train_final = Y_train_clean


# # Train the model
# pizza_pred = LinearRegression()
# pizza_pred.fit(X_train_final, Y_train_final)

# print("🍕 Model trained successfully!")


# os.system('pause')
# os.system('cls')  # Mac


# # Get number of orders
# order_count = int(input("Enter number of orders: "))

# # Initialize empty lists to store all orders
# dist = []
# piz = []
# day = []
# weather = []
# traffic = []

# print("\n🍕 NEW ORDERS COMING IN!")
# for i in range(order_count):
#     print(f"\n--- Order {i+1} ---")
#     dist.append(float(input("Enter distance in miles: ")))
#     piz.append(int(input("Enter number of pizzas: ")))
#     day.append(int(input("Enter day of week (1-7): ")))
#     weather.append(int(input("Enter weather (1-4): ")))
#     traffic.append(int(input("Enter traffic level (1-3): ")))

# # Create DataFrame with all orders
# new_orders = pd.DataFrame({
#     'distance_miles': dist,
#     'pizza_count': piz,
#     'day_of_week': day,
#     'weather': weather,
#     'traffic_level': traffic
# })

# predictions = pizza_pred.predict(new_orders)

# days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
# weather_names = ['Sunny', 'Cloudy', 'Rainy', 'Snowy']
# traffic_names = ['Light', 'Medium', 'Heavy']

# print("\n📞 Customer Calls:")
# print("=" * 50)

# for i in range(len(new_orders)):
#     order = new_orders.iloc[i]
#     day_name = days[int(order['day_of_week'])-1]
#     weather_name = weather_names[int(order['weather'])-1]
#     traffic_name = traffic_names[int(order['traffic_level'])-1]

#     print(f"\n📞 Order {i+1}:")
#     print(f"   📍 Distance: {order['distance_miles']} miles")
#     print(f"   🍕 Pizzas: {int(order['pizza_count'])}")
#     print(f"   📅 Day: {day_name}")
#     print(f"   🌤️  Weather: {weather_name}")
#     print(f"   🚦 Traffic: {traffic_name}")
#     print(f"   ⏰ Predicted time: {predictions[i]:.1f} minutes")
#     print(f"   💬 'Your pizza will arrive in about {int(predictions[i])} minutes!'")
#     print("-" * 30)


# # # Optional: Evaluate the model on test data
# Y_test_clean = pd.to_numeric(Y_test, errors='coerce')
# test_valid_indices = ~Y_test_clean.isna()
# X_test_final = X_test[test_valid_indices]
# Y_test_final = Y_test_clean[test_valid_indices]

# if len(X_test_final) > 0:
#     test_score = pizza_pred.score(X_test_final, Y_test_final)
#     print(f"\n📊 Model R² Score on test data: {test_score:.3f}")
# else:
#     print("\n⚠️ No valid test data available for evaluation")

# jb.dump(pizza_pred, "Model.pkl")
# print('Model saved to Model.pkl successfully.')


import os
import numpy as np
import joblib as jb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_squared_error

pizza_data = pd.read_csv("pizza_delivery_data.csv")
features = ["distance_miles", "pizza_count", "day_of_week", "weather", "traffic_level"]
X = pizza_data[features]
Y = pizza_data["delivery_time"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print(f"\nData Training on {len(X_train)} deliveries")
print(f"Data Testing on {len(X_test)} deliveries")

# Clean the target variable
Y_train_clean = pd.to_numeric(Y_train, errors="coerce")
nan_count = Y_train_clean.isna().sum()
print(f"Found {nan_count} NaN values in Y_train")

# Remove rows with NaN values from both X_train and Y_train_clean
if nan_count > 0:
    # Get indices where Y_train_clean is not NaN
    valid_indices = ~Y_train_clean.isna()  # '~' is used as a not operator in pandas

    # Filter both X and Y to remove NaN rows
    X_train_final = X_train[valid_indices]
    Y_train_final = Y_train_clean[valid_indices]

    print(f"After cleaning: Training on {len(X_train_final)} deliveries")
else:
    X_train_final = X_train
    Y_train_final = Y_train_clean

# Train the Decision Tree model
pizza_pred = DecisionTreeRegressor(
    max_depth=10,  # Limit tree depth to prevent overfitting
    min_samples_split=20,  # Minimum samples required to split a node
    min_samples_leaf=10,  # Minimum samples required at leaf node
    random_state=42,  # For reproducible results
)
pizza_pred.fit(X_train_final, Y_train_final)

print("🌳 Decision Tree model trained successfully!")

# Optional: Visualize the decision tree (first few levels)
plt.figure(figsize=(20, 10))
plot_tree(
    pizza_pred,
    feature_names=features,
    max_depth=3,  # Only show first 3 levels for readability
    filled=True,
    fontsize=10,
)
plt.title("Pizza Delivery Time Decision Tree (First 3 Levels)")
plt.savefig("decision_tree_visualization.png", dpi=300, bbox_inches="tight")
plt.show()

# Display feature importance
feature_importance = pd.DataFrame(
    {"feature": features, "importance": pizza_pred.feature_importances_}
).sort_values("importance", ascending=False)

print("\n🎯 Feature Importance:")
print("=" * 30)
for idx, row in feature_importance.iterrows():
    print(f"{row['feature']:15}: {row['importance']:.3f}")

os.system("pause")
os.system("cls")  # Clear screen

# Get number of orders
order_count = int(input("Enter number of orders: "))

# Initialize empty lists to store all orders
dist = []
piz = []
day = []
weather = []
traffic = []

print("\n🍕 NEW ORDERS COMING IN!")
for i in range(order_count):
    print(f"\n--- Order {i+1} ---")
    dist.append(float(input("Enter distance in miles: ")))
    piz.append(int(input("Enter number of pizzas: ")))
    day.append(int(input("Enter day of week (1-7): ")))
    weather.append(int(input("Enter weather (1-4): ")))
    traffic.append(int(input("Enter traffic level (1-3): ")))

# Create DataFrame with all orders
new_orders = pd.DataFrame(
    {
        "distance_miles": dist,
        "pizza_count": piz,
        "day_of_week": day,
        "weather": weather,
        "traffic_level": traffic,
    }
)

predictions = pizza_pred.predict(new_orders)

days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
weather_names = ["Sunny", "Cloudy", "Rainy", "Snowy"]
traffic_names = ["Light", "Medium", "Heavy"]

print("\n📞 Customer Calls:")
print("=" * 50)

for i in range(len(new_orders)):
    order = new_orders.iloc[i]
    day_name = days[int(order["day_of_week"]) - 1]
    weather_name = weather_names[int(order["weather"]) - 1]
    traffic_name = traffic_names[int(order["traffic_level"]) - 1]

    print(f"\n📞 Order {i+1}:")
    print(f"   📍 Distance: {order['distance_miles']} miles")
    print(f"   🍕 Pizzas: {int(order['pizza_count'])}")
    print(f"   📅 Day: {day_name}")
    print(f"   🌤️  Weather: {weather_name}")
    print(f"   🚦 Traffic: {traffic_name}")
    print(f"   ⏰ Predicted time: {predictions[i]:.1f} minutes")
    print(f"   💬 'Your pizza will arrive in about {int(predictions[i])} minutes!'")
    print("-" * 30)

# Evaluate the model on test data
Y_test_clean = pd.to_numeric(Y_test, errors="coerce")
test_valid_indices = ~Y_test_clean.isna()
X_test_final = X_test[test_valid_indices]
Y_test_final = Y_test_clean[test_valid_indices]

if len(X_test_final) > 0:
    test_predictions = pizza_pred.predict(X_test_final)
    test_score = r2_score(Y_test_final, test_predictions)
    test_mse = mean_squared_error(Y_test_final, test_predictions)
    test_rmse = np.sqrt(test_mse)

    print(f"\n📊 Model Performance on Test Data:")
    print(f"   R² Score: {test_score:.3f}")
    print(f"   RMSE: {test_rmse:.2f} minutes")
    print(
        f"   Mean Absolute Error: {np.mean(np.abs(Y_test_final - test_predictions)):.2f} minutes"
    )
else:
    print("\n⚠️ No valid test data available for evaluation")

# Save the model
jb.dump(pizza_pred, "Model.pkl")
print("\n🌳 Decision Tree model saved to DecisionTree_Model.pkl successfully.")

# Optional: Display some decision rules (interpretability feature of decision trees)
print(f"\n🌳 Decision Tree Information:")
print(f"   Tree depth: {pizza_pred.get_depth()}")
print(f"   Number of leaves: {pizza_pred.get_n_leaves()}")
print(f"   Number of features used: {pizza_pred.n_features_in_}")
