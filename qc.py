# C (i)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 5.0)
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import PolynomialFeatures
df = pd.read_csv("week2.csv", comment = '#')

#feature values
X1 = df.iloc[:, 0]
X2 = df.iloc[:, 1]
X = np.column_stack((X1, X2))
y = df.iloc[:, 2]
#print(y)

#adding the square of each feature
X1_new = np.square(X1)
#print("X1\n", X1)	#1st feature = 0.59
#print("New X1 squared: \n", X1_new)	#0.59^2 = 0.3481

X2_new = np.square(X2)
#print("X2\n", X2)	#1st feature = 0.61
#print("New X2 squared: \n", X2_new) #0.61^2 = 0.3721

X_new = np.column_stack((X1, X1_new, X2, X2_new))
#print("X_new:\n", X_new)

#distinguish +1 and -1
plus = df.loc[y == 1]	#+1
minus = df.loc[y == -1]	#-1

#using sklearn
model = LogisticRegression(penalty='none', solver='lbfgs')
model.fit(X_new, y)
intercept = model.intercept_
coefficient = model.coef_
prediction = model.predict(X_new)

print("Intercept:", intercept)
print("Coefficients:", coefficient)

plt.scatter(X1, prediction, color = 'orange', label = 'X1 Prediction')
plt.scatter(X2, prediction, color = 'black', label = 'X2 Prediction')

plt.scatter(plus.iloc[:, 0], plus.iloc[:, 1], s = 25, color = 'blue', label = '+1', marker = '+')
plt.scatter(minus.iloc[:, 0], minus.iloc[:, 1], s = 25, color = 'green', label = '-1', marker = 'o')
plt.gca().set_title("Assignment 2", color = 'black')
plt.xlabel("X1", color = 'blue'); plt.ylabel("X2", color = 'blue')
plt.legend()
plt.show()