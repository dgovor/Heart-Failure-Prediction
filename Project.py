# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMClassifier

data_set = pd.read_csv('heart.csv')

missing = data_set.isna().sum()*100/data_set.shape[0]

le = LabelEncoder()
data_set['Sex']=le.fit_transform(data_set['Sex'])
data_set['RestingECG']=le.fit_transform(data_set['RestingECG'])
data_set['ChestPainType']=le.fit_transform(data_set['ChestPainType'])
data_set['ExerciseAngina']=le.fit_transform(data_set['ExerciseAngina'])
data_set['ST_Slope']=le.fit_transform(data_set['ST_Slope'])

correlation = data_set.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, vmin=-1, vmax=1, cbar=True, square=True, annot=True, fmt='.1f', cmap='vlag')
plt.show()

X = data_set.drop(['HeartDisease'], axis=1)
Y = data_set['HeartDisease']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=34, stratify=Y)

ct = ColumnTransformer([('MinMax', MinMaxScaler(), ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak'])], remainder='passthrough')

ct.fit(X_train)
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)

model = LGBMClassifier(learning_rate=0.012, n_estimators=500, random_state=34)
model.fit(X_train,Y_train)

predictions = model.predict(X_test)
error = metrics.accuracy_score(Y_test, predictions)

plt.figure(figsize=(5,5))
con_mat = plt.subplot()
sns.heatmap(metrics.confusion_matrix(Y_test,predictions), annot=True, cmap='Blues', cbar=False)
con_mat.set_xlabel('Predicted')
con_mat.set_ylabel('Actual'); 
con_mat.set_xticklabels(['Negative', 'Positive'], ha='center')
con_mat.set_yticklabels(['Negative', 'Positive'], va='center')
con_mat.set_title('Confusion Matrix'); 
plt.show()