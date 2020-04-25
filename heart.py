import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
import pickle
warnings.filterwarnings('ignore')

# Other libraries
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
dataset=pd.read_csv("C:/Users/mahsa/OneDrive/Desktop/image_python_machine_deep/data/heart-disease-dataset/heart.csv")
y = dataset['target']
X = dataset.drop(['target'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
pickle.dump(dt_classifier, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[52, 1, 0,125,212,0,1,168,0,1,2,2,3]]))