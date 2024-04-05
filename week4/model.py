#Training ML Model

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
df = pd.read_csv('/Users/sandeepbansal/Downloads/50_Startup.csv')
df.head()
x = df.iloc[:, :-1].values
y = df.iloc[:, 3].values
regressor = LinearRegression()
regressor.fit(x,y)
pickle.dump(regressor, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[16000, 135000, 450000]]))
