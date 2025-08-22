# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
# Date:19-08-2025
### NAME:HYCINTH D
### REGISTER NO:212223240055
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
### A - LINEAR TREND ESTIMATION
```PYTHON
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

file_path = "Amazon.csv"
data = pd.read_csv(file_path)

data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values(by="Date")
X = np.arange(len(data)).reshape(-1, 1) 
y = data["Volume"].values

linear_model = LinearRegression()
linear_model.fit(X, y)

data["Linear_Trend"] = linear_model.predict(X)

plt.figure(figsize=(12, 6))
plt.plot(data["Date"], y, label="Original Data", marker="o", markersize=4, linestyle="None")
plt.plot(data["Date"], data["Linear_Trend"], color="yellow", label="Linear Trend", linewidth=2)
plt.title("Linear Trend Estimation - Amazon Volume")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.legend()
plt.grid(True)
plt.show()

```
### B- POLYNOMIAL TREND ESTIMATION
```PYTHON
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

file_path = "Amazon.csv"
data = pd.read_csv(file_path)

data["Date"] = pd.to_datetime(data["Date"])

data = data.sort_values(by="Date")

X = np.arange(len(data)).reshape(-1, 1) 
y = data["Volume"].values

poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

data["Poly_Trend"] = poly_model.predict(X_poly)

plt.figure(figsize=(12, 6))
plt.plot(data["Date"], y, label="Original Data", marker="o", alpha=0.6, markersize=4, linestyle="None")
plt.plot(data["Date"], data["Poly_Trend"], color="red", label="Polynomial Trend (Degree 2)", linewidth=2)
plt.title("Polynomial Trend Estimation - Amazon Volume")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.legend()
plt.grid(True)
plt.show()

```
### OUTPUT
### A - LINEAR TREND ESTIMATION
<img width="1115" height="601" alt="image" src="https://github.com/user-attachments/assets/4c2184b8-d592-42f0-9712-fa011dc4ddb9" />

### B- POLYNOMIAL TREND ESTIMATION

<img width="1133" height="600" alt="image" src="https://github.com/user-attachments/assets/3c18a0de-8591-4c6a-bd92-d2fa64d106ac" />


### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
