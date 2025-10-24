from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

  # Define data (x now matches y's length)
x = np.array([[1],[2],[3],[4],[5]]) # Reshape for sklearn
y = np.array([2, 4, 6, 8, 10])
  # Create and fit the model (replaces modlex/modley)
model = LinearRegression()
model.fit(x, y)
  # Predict y for the x values
x_pred = x  # Use the same x for prediction
y_pred = model.predict(x_pred)
  # Plot
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x_pred, y_pred, color='red', label='Fitted Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Fit')
plt.legend()
plt.show()