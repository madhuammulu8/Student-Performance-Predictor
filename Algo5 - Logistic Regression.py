from sklearn.preprocessing import StandardScaler
import numpy as np
import random as rd
import math
import operator
import pandas as pd
import sklearn as sk
import sklearn.neighbors as ng
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn import metrics
from sklearn.metrics import accuracy_score

#LOGISTIC REGRESSION
class LogisticRegression(object):
    #Constructor, Setting by default values for Learning Rate (Alpha) and Number of Iterations if not passed manually
    def __init__(self, learning_rate=0.01, number_of_iterations=100):                           
        self.number_of_iterations = number_of_iterations 
        self.learning_rate = learning_rate   
    
    #function to train the model
    def fit(self, X, y): 
        self.theta = []
        self.cost = []
        train = np.insert(X, 0, 1, axis=1)
        m = len(y)
        unique = np.unique(y)
        for i in unique: 
            y_onevsall = np.where(y == i, 1, 0)
            theta = np.zeros(train.shape[1])
            cost = []
            for _ in range(self.number_of_iterations):
                h = self.sigmoidFunction(train.dot(theta))
                theta = self.computeGradient(train,h,theta,y_onevsall,m)
                cost.append(self.computeCost(h,theta,y_onevsall)) 
            self.theta.append((theta, i))
            self.cost.append((cost,i))
        return self

    #Function to calculate sigmoid value
    def sigmoidFunction(self, x): 
      return 1 / (1 + np.exp(-x))
        
    #Function to calculate cost
    def computeCost(self,h,theta, y): 
        len_y = len(y)
        cost = (1 / len_y) * (np.sum(-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))))
        return cost

    #Function to calculate predicted value
    def computePrediction(self, X): 
        inputs = np.insert(X, 0, 1, axis=1)
        predicted_values = [max((self.sigmoidFunction(i.dot(theta)), c) for theta, c in self.theta)[1] for i in inputs]
        return predicted_values 

    #Function to calculate gradient
    def computeGradient(self,X,h,theta,y,m): 
        gradient_value = np.dot(X.T, (h - y)) / m
        theta = theta - self.learning_rate * gradient_value
        return theta

    #Funtion to return the Accuracy
    def computeScore(self,X, y): 
        score = sum(self.computePrediction(X) == y) / len(y)
        return score   
df = pd.read_csv("student_data.csv")
df["passed"] = df["passed"].astype('category')
# df.dtypes
df["passed"] = df["passed"].cat.codes

outcomes = df['passed']
data = df.drop('passed', axis = 1)

data = data.dropna()
data = data.apply(lambda col: pd.factorize(col, sort=True)[0])
scaler = preprocessing.MinMaxScaler()
scaler.fit(data)
scaled = scaler.transform(data)
train = pd.DataFrame(scaled, columns=data.columns)


scaler = StandardScaler()
train= scaler.fit_transform(train)
X_train,X_test,y_train,y_test = train_test_split(train, outcomes,stratify=outcomes,test_size=0.20)
LR = LogisticRegression(number_of_iterations=300).fit(X_train, y_train)
predictions = LR.computePrediction(X_test)
score1 = LR.computeScore(X_test,y_test)
print("The accuracy of the model is ",score1*100)
print(metrics.confusion_matrix(y_test, predictions, labels=[0,1]))

print(metrics.classification_report(y_test, predictions, labels=[0,1]))