import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('data\Irisdataset.csv')
print(data.head())
print("Missing values:")
print(data.isnull().sum())
data.dropna(inplace=False)
le = LabelEncoder()
data["Species"]=le.fit_transform(data["Species"])
le.classes_
data.dropna(inplace=True)
selected_features = ("SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm")
X = data['SepalLengthCm']
y = data['Species']
x_train,x_test,y_train,y_test=train_test_split(X.values.reshape(-1,1),y,test_size=0.2,random_state=40)
model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy}")
plt.scatter(x_test, y_test, color='green')
plt.plot(np.sort(x_test,axis=0), model.predict_proba(np.sort(x_test,axis=0))[:,1], color='blue',linewidth=3)
plt.xlabel('SepalLengthCm')
plt.ylabel('Species')
plt.title('Logistic regression Decision Boundary')
plt.show()