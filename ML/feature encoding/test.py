import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import math
import pymysql
from sklearn import preprocessing


pd.set_option('display.max_column', None)

# MYSQL connection
# dbcon = pymysql.connect(host="localhost", user="root", password="4v0hs0K8oo2", database="intro")
# df = pd.read_sql_query("""SELECT * FROM telco_customer_churn""", dbcon, parse_dates=True)

# x = df.head()
# print(x)

df = pd.read_csv('./Telco-Customer-Churn.csv')
print(df.head(5))
print(df.value_counts())

# Handle null values
mode = df.gender.mode()
df["gender"] = df["gender"].fillna(mode)
print(df.info())

# Label encoding
le = preprocessing.LabelEncoder()
df["Gender_Label"] = le.fit_transform(df.gender.values)
print(df.Gender_Label.value_counts())

# One hot encoding
one_hot = pd.get_dummies(df["PaymentMethod"])
print(one_hot.value_counts())