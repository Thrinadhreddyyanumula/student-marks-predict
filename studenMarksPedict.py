# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 08:00:25 2023

@author: 91955
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as
plt
data=pd.read_csv(r"C:\Users\91955\AppData\Local\Temp\Rar$DIa13168.19793/student_info.
csv")
data.isnull().sum()
data.mean()
data1=data.fillna(data.mean())
data1.isnull().su
m()
X=data1.iloc[:,-1]
y=data1.iloc[:,0]
X=data1.drop("student_marks",axis="c
olumns")
y=data1.drop("study_hours",axis="columns")
from
sklearn.model_selection import
train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y,
random_state=0,test_size=0.2,)
from sklearn.linear_model import
LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
lr.coef_
lr.intercept_
y_pr
ed=lr.predict(X_test)
pd.DataFrame(np.c_[X_test,y_test,y_pred],columns=["student_marks&qu
ot;,"study_hours","pred_marks"])
import joblib
joblib.dump(lr,
"student_marks_info.pkl")
import joblib
joblib.load("student_marks_info.pkl")
lr.predict([[5]])[0][0]
lr.predict([[18]])[
0][0]
lr.predict([[25]])[0][0]
import joblib
joblib.dump(lr,
"student_marks_info.pkl")
import numpy as np
import pandas as pd
from flask
import Flask, request, render_template
import joblib
app = Flask(__name__)
model =
joblib.load(r"C:\Users\91955\student_marks_info.pkl")
df =
pd.DataFrame()
@app.route('/')
def home():
 return
render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
 
global data
 
 input_features = [int(x) for x in request.form.values()]
features_value = np.array(input_features)
 
 #validate input hours
 if
input_features[0] <0 or input_features[0] >24:
 return
render_template('index.html', prediction_text='Please enter valid hours between 1 to 24 if you
live on the Earth')
 
 output =
model.predict([features_value])[0][0].round(2)
 # input and predicted value store in df
then save in csv file
 df= pd.concat([df,pd.DataFrame({'Study
Hours':input_features,'Predicted Output':[output]})],ignore_index=True)
 print(df) 
 
df.to_csv('smp_data_from_app.csv')
 return render_template('index.html',
prediction_text='You will get [{}%] marks, when you do study [{}] hours per day
'.format(output, int(features_value[0])))
if __name__ == "__main__":
 
app.run(host='127.0.0.1', port=5000)
Powered by TCPDF (www.tcpdf.org)