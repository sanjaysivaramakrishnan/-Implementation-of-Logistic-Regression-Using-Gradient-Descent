# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.

2.Load the dataset.

3.Define X and Y array.

4.Define a function for costFunction,cost and gradient.

5.Define a function to plot the decision boundary

6.Define a function to predict the Regression value.



## Program:

```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by   : Sanjay Sivaramakrishnan M
RegisterNumber :  212223240151
```

```
import pandas as pd
import numpy as np
df=pd.read_csv("Home/.../.../.../Placement_Data.csv")
df
df=df.drop('sl_no',axis=1)
df=df.drop('salary',axis=1)
df.head()
df["gender"]=df["gender"].astype("category")
df["ssc_b"]=df["ssc_b"].astype("category")
df["hsc_b"]=df["hsc_b"].astype("category")
df["hsc_s"]=df["hsc_s"].astype("category")
df["degree_t"]=df["degree_t"].astype("category")
df["workex"]=df["workex"].astype("category")
df["specialisation"]=df["specialisation"].astype("category")
df["status"]=df["status"].astype("category")
df.dtypes
df["gender"]=df["gender"].cat.codes
df["ssc_b"]=df["ssc_b"].cat.codes
df["hsc_b"]=df["hsc_b"].cat.codes
df["hsc_s"]=df["hsc_s"].cat.codes
df["degree_t"]=df["degree_t"].cat.codes
df["workex"]=df["workex"].cat.codes
df["specialisation"]=df["specialisation"].cat.codes
df["status"]=df["status"].cat.codes
df
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
y
theta = np.random.randn(x.shape[1])
Y=y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,x,Y):
    h=sigmoid(x.dot(theta))
    return -np.sum(Y*np.log(h)+(1-Y)*np.log(1-h))

def gradient_descent(theta,x,Y,alpha,num_iterations):
    m=len(Y)
    for i in range(num_iterations):
        h=sigmoid(x.dot(theta))
        gradient=x.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta

theta=gradient_descent(theta,x,Y,alpha=0.01,num_iterations=1000)
def predict(theta,x):
    h=sigmoid(x.dot(theta))
    y_pred=np.where(h>0.5,1,0)
    return y_pred
y_pred=predict(theta,x)
accuracy=np.mean(y_pred.flatten()==Y)
print("Accuracy:",accuracy)
print(y_pred)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

## Output:
![image](https://github.com/user-attachments/assets/849d9352-a5c2-46a9-9370-ad417b831406)
![image](https://github.com/user-attachments/assets/d7384f3d-0552-436b-83c4-d824598e8792)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

