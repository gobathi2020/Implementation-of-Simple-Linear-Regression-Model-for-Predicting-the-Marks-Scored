# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Step 1: Start

Step 2: Import the standard Libraries.

Step 3: Set variables for assigning dataset values.

Step 4: Import linear regression from sklearn.

Step 5: Assign the points for representing in the graph.

Step 6: Predict the regression for marks by using the representation of the graph.

Step 7: Compare the graphs and hence we obtained the linear regression for the given datas.

Step 8: End

## Program:

/* Program to implement the simple linear regression model for predicting the marks scored.

Developed by: GOBATHI P

Register Number:212222080017 */

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='purple')
plt.plot(x_train,regressor.predict(x_train),color='yellow')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```

## Output:
![image](https://github.com/user-attachments/assets/716a535c-65cd-49bc-afb4-252751b4f9a2)

![image](https://github.com/user-attachments/assets/b076caa5-53e0-4612-b3d8-b4f66a9cfa3a)

![image](https://github.com/user-attachments/assets/89e5c020-935d-4f5e-ba77-55885d1315e6)

![image](https://github.com/user-attachments/assets/9865a9c1-f170-4efc-8dc9-bbff590bdaa5)

![image](https://github.com/user-attachments/assets/a336c03e-8105-4f3c-8b01-f5b05816b34f)

![image](https://github.com/user-attachments/assets/60d4e53a-a032-49f7-bbcd-1fc67cce3062)

![image](https://github.com/user-attachments/assets/12cdcb31-3a0c-401d-9dc1-eca3865f5c09)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
