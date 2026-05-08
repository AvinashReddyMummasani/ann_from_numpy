from ann import Neural_network
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from sklearn.preprocessing import LabelEncoder,StandardScaler
import pandas as pd
import numpy as np

#start

df=pd.read_csv('diabetes.csv')
X=df.drop('Outcome',axis=1).values
se=StandardScaler()
X=se.fit_transform(X)
y=df['Outcome'].values

#end

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=42)

ann=Neural_network([X.shape[1],3,2,1],['relu','relu','relu'],'bce',0.01)
ann.fit(X_train,y_train,epochs=5)
y_pred=ann.predict(X_test)
# print("r2score :",r2_score(y_test,y_pred))
y_pred=np.where(y_pred>=0.5,1,0)
print("accuracy :",accuracy_score(y_test,y_pred))


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import tensorflow
from tensorflow.keras.layers import Dense,Input,BatchNormalization,Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
model=Sequential()
model.add(Input(shape=(X.shape[1],)))
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
print(model.summary())
model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='binary_crossentropy'
)
model.fit(X_train,y_train,epochs=5)
y_pred=model.predict(X_test)
# print("r2score :",r2_score(y_test,y_pred))
y_pred=np.where(y_pred>=0.5,1,0)
print("accuracy :",accuracy_score(y_test,y_pred))