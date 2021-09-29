import  numpy as np
import pandas as pd
from sklearn.ensemble import  ExtraTreesRegressor
from  sklearn.model_selection import train_test_split

import  warnings
warnings.filterwarnings('ignore')

import pickle

Data = pd.read_csv('Resturant_clean_data.csv')
# spliting in dependent and independent set
X = Data.drop('rate', axis=1)
Y = Data['rate']

#creating train,test split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, random_state=10, test_size=.4)

# creating model
ext = ExtraTreesRegressor(min_samples_leaf=1e-05, n_estimators=300,random_state= 250)
ext.fit(X_train,Y_train)

y_pred = ext.predict(X_test)
print(y_pred)

# generating pickel file
pickle.dump(ext,open('ExtraTreesregression_model.pkl','wb'))
model = pickle.load(open('ExtraTreesregression_model.pkl','rb'))
