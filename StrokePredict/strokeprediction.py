import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read Dataset
df=pd.read_csv("/content/sample_data/healthcare-dataset-stroke-data.csv")
df.info()

"""Preprocessing data"""

# Droping Nan values
df.dropna(inplace=True)

#Droping id column
df.drop(columns=["id"],inplace=True)

df.head(2)

#Checking what are the different values present in each non numeric values
print(df['gender'].value_counts())
print(df['ever_married'].value_counts())
print(df['work_type'].value_counts())
print(df['Residence_type'].value_counts())
print(df['smoking_status'].value_counts())

# Transforming categorical data to numerical data
categorical_Values=["gender","ever_married","work_type","Residence_type","smoking_status"]
inputdata=pd.get_dummies(df,columns=categorical_Values,drop_first=True)

inputdata.head(2)

inputdata.info()

"""Data Spread Analyze"""

inputdata.hist(figsize=(15,8))

"""Correlation Analysis"""

plt.figure(figsize=(13,6))
sns.heatmap(inputdata.corr(),annot=True,cmap="YlGnBu")

"""Train Test Split"""

X=inputdata.drop(columns=["stroke"])
y=inputdata["stroke"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

X_train.count()

"""Scaling Data"""

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#checkimg number of columns
len(X_train_scaled[0])

"""Creating ANN model for prediction of stroke"""

import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()
model.add(Dense(7,activation="sigmoid",input_dim=16))
model.add(Dense(3,activation="sigmoid",input_dim=7))
model.add(Dense(2,activation="sigmoid",input_dim=3))
model.add(Dense(1,activation="sigmoid"))

model.summary()

#Compilig and training Model
model.compile(loss="binary_crossentropy",optimizer="Adam")
model.fit(X_train_scaled,y_train,epochs=10)

y_out=model.predict(X_test_scaled)

y_out

#Since sigmoid function has been used, therefore values lie between 0 and 1
# Considering value above 0.5 is 1 and below 0.5 is 0

y_pred=np.where(y_out>0.5,1,0)

y_pred

"""Calculating Accuracy of the model"""

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

"""Accuracy of the model is 94%"""

