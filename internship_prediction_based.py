import pandas as pd
df= pd.read_csv("internship_prediction_based.csv")

X = df.iloc[:, [2, 3]].values
y = df.iloc[:, 4].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,
                            test_size=.25,random_state=42)
from sklearn.preprocessing import MinMaxScaler
ms=MinMaxScaler()
x_train=ms.fit_transform(x_train)
x_test=ms.transform(x_test)

from sklearn.svm import SVC
model=SVC(kernel='linear',random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

from sklearn.metrics import accuracy_score
print("Linear Accuracy: ",accuracy_score(y_test,y_pred))

model=SVC(kernel='rbf',random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

from sklearn.metrics import accuracy_score
print("RBF Accuracy: ",accuracy_score(y_test,y_pred))

model=SVC(kernel='poly',random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

from sklearn.metrics import accuracy_score
print("Poly Accuracy: ",accuracy_score(y_test,y_pred))

model=SVC(kernel='sigmoid',random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

from sklearn.metrics import accuracy_score
print("Sigmoid Accuracy: ",accuracy_score(y_test,y_pred))

from sklearn.naive_bayes import GaussianNB
gb=GaussianNB()
gb.fit(x_train,y_train)
y_pred=gb.predict(x_test)
from sklearn.metrics import accuracy_score
print("NB accuracy=",accuracy_score(y_test,y_pred))