# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix



cancer = load_breast_cancer()

print (cancer['target'])

print(cancer['feature_names'])

df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))

sns.pairplot(df_cancer, hue='target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])

sns.countplot(df_cancer['target'], label = "Count")

sns.scatterplot(x = 'mean area', y= 'mean smoothness', hue = 'target', data = df_cancer)

plt.figure(figsize=(20,10))
sns.heatmap(df_cancer.corr(), annot=True)

X = df_cancer.drop(['target'], axis=1)
y = df_cancer['target']

input_train, input_test, label_train, label_test = train_test_split(X, y, test_size=0.2, random_state=5)

svc_model = SVC()
svc_model.fit(input_train, label_train)

y_predict = svc_model.predict(input_test)
cm = confusion_matrix(label_test, y_predict)

sns.heatmap(cm, annot=True)
plt.show()

