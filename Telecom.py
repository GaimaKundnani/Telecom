import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,precision_score,f1_score,recall_score
import seaborn as sns
#Load Dataset
df=pd.read_csv(r"/Users/Garima/Desktop/Python/Project-5/telecomes.csv")
print(df)
#Infomation
print(df.info())
#Description
print(df.describe())
#Scatter Plot
fig=plt.figure(figsize=(5,3))
plt.scatter(df.MonthlyCharges ,df.Churn)
plt.xlabel("MonthlyCharges")
plt.ylabel("Churn")
plt.show()
#Split Dataset
X_train,X_test,Y_train,Y_test=train_test_split(df[["MonthlyCharges"]],df[["Churn"]],test_size=0.3,random_state=1)
print(X_train)
print(Y_train)
model=LogisticRegression()
model.fit(X_train,Y_train)
pre=model.predict(X_test)
#New Dataframe
result = pd.DataFrame({'tenure': X_test['MonthlyCharges'], 'Actual_Result': Y_test['Churn'], 'Prediction': pre})
print(result)
#Creating the Confusion matrix 
cm = confusion_matrix(Y_test, pre)
print("Confusion Matrix:")
print(cm)
#Accuracy
print("Accuracy:",model.score(X_test,Y_test)*100)
#Heat Map
plt.figure(figsize=(5,5))
sns.heatmap(cm,annot=True)
plt.xlabel("Prediction")
plt.ylabel("Truth")
plt.show()
#Precision
precision = precision_score(Y_test, pre,pos_label="Yes")
print("Precision:", precision)
# F1 Score
f1 = f1_score(Y_test, pre,pos_label="Yes")
print("F1 Score:", f1)

#Recall
recall = recall_score(Y_test, pre, pos_label="Yes")
print("Recall (Sensitivity):", recall)

#Calculate Revenue
df['Revenue'] = df['MonthlyCharges'] * df['tenure']
print(df[['MonthlyCharges', 'tenure', 'Revenue']])
