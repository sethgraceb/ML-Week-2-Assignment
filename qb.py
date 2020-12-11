# B(i) C=0.001
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 5.0)
from sklearn.linear_model import LogisticRegression 
df = pd.read_csv("week2.csv", comment = '#')

#feature values
X1 = np.array(df.iloc[:, 0])
X2 = np.array(df.iloc[:, 1])
X = np.column_stack((X1, X2))
y = np.array(df.iloc[:, 2])
#print(X1)
#print(X2)
#print(y)

#distinguish +1 and -1
plus = df.loc[y == 1]	#+1
minus = df.loc[y == -1]	#-1

#using LinearSVC
from sklearn.svm import LinearSVC
#using sklearn
model = LogisticRegression(penalty='none', solver='lbfgs')
model = LinearSVC(C=0.001).fit(X, y)    #0.001 #1.0 #1000
intercept = model.intercept_
coefficient = model.coef_
prediction = model.predict(X)

print("Intercept:", intercept)
print("Coefficients:", coefficient)

#decision boundary
w = coefficient[0]
a = -w[0] / w[1]
db = np.linspace(-1, 1)
db2 = a * db - (intercept[0] / w[1])
plt.plot(db, db2, color = 'black', label = 'Decision Boundary')

plt.scatter(X2, prediction, color = 'red', marker = 'x', label = 'Prediction')	#x2 works #y gives 4 x's on each side of the plane

plt.scatter(plus.iloc[:, 0], plus.iloc[:, 1], s = 25, color = 'blue', label = '+1', marker = '+')
plt.scatter(minus.iloc[:, 0], minus.iloc[:, 1], s = 25, color = 'green', label = '-1', marker = 'o')
plt.gca().set_title("Assignment 2", color = 'black')
plt.xlabel("X1", color = 'blue'); plt.ylabel("X2", color = 'blue')
plt.legend()
plt.show()