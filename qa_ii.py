import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 5.0)
from sklearn.linear_model import LogisticRegression 
df = pd.read_csv("week2.csv", comment = '#')

#feature values
X1 = df.iloc[:, 0]
X2 = df.iloc[:, 1]
X = np.column_stack((X1, X2))
y = df.iloc[:, 2]

#distinguish +1 and -1
plus = df.loc[y == 1]	#+1
minus = df.loc[y == -1]	#-1

#using sklearn
model = LogisticRegression(penalty='none', solver='lbfgs')
model.fit(X, y)
intercept = model.intercept_
coefficient = model.coef_

print("Intercept:", intercept)
print("Coefficients:", coefficient)

plt.scatter(plus.iloc[:, 0], plus.iloc[:, 1], s = 25, color = 'blue', label = '+1', marker = '+')
plt.scatter(minus.iloc[:, 0], minus.iloc[:, 1], s = 25, color = 'green', label = '-1', marker = 'o')
plt.gca().set_title("Assignment 2", color = 'black')
plt.xlabel("X1", color = 'blue'); plt.ylabel("X2", color = 'blue')
plt.legend()
plt.show()