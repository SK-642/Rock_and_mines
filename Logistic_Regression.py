import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sonar_data=pd.read_csv("D:/Project/copy_of_sonar_data.csv", header=None)
#first 5 rows are printed)
print(sonar_data.head())
#gives shape of data
print(sonar_data.shape)
#describe gives statistical data
print(sonar_data.describe())  
#gives number of rocks(R) and mines(M)
print(sonar_data[60].value_counts())
print(sonar_data.groupby(60).mean())

#separating data and labels
x=sonar_data.drop(columns=60, axis=1)
y=sonar_data[60]
#Training and test data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,stratify=y,random_state=1)
print(x.shape,x_train.shape,x_test.shape)
#model_training = Logistic Regression
model = LogisticRegression()
#training the LR with training data
model.fit(x_train,y_train)
#model_Evaluation
#accuracy on training data
x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print("Accuracy on training data: ", training_data_accuracy)

#accuracy on test data

x_test_prediction=model.predict(x_test)
training_data_accuracy=accuracy_score(x_test_prediction,y_test)
print("Accuracy on test data: ", training_data_accuracy)

#making a predictive system
input_data=(0.0664,0.0575,0.0842,0.0372,0.0458,0.0771,0.0771,0.1130,0.2353,0.1838,0.2869,0.4129,0.3647,0.1984,0.2840,0.4039,0.5837,0.6792,0.6086,0.4858,0.3246,0.2013,0.2082,0.1686,0.2484,0.2736,0.2984,0.4655,0.6990,0.7474,0.7956,0.7981,0.6715,0.6942,0.7440,0.8169,0.8912,1.0000,0.8753,0.7061,0.6803,0.5898,0.4618,0.3639,0.1492,0.1216,0.1306,0.1198,0.0578,0.0235,0.0135,0.0141,0.0190,0.0043,0.0036,0.0026,0.0024,0.0162,0.0109,0.0079)
#changing data to numpy array
input_data_as_numpy_array = np.asarray(input_data)
#reshaping the numpy array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped) 
print(prediction)

if prediction[0] == 'R':
    print("the object is rock")
else:
    print("it is a Mine")

