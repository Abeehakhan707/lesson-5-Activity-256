import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns   
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('Churn_Modelling.csv')
df.head()
df.info()
df.describe()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['Gender'] = le.fit_transform(df['Gender'])
df['Geography'] = le.fit_transform(df['Geography'])
df.head()

df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
df.head()

y = df.pop('Exited')
x = df

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
x_train