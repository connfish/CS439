from sklearn.linear_model import LogisticRegression
import pandas as pd
df = pd.read_csv('data')
X = pd.get_dummies(df[['age','sex','edu']], drop_first=True)
y = (df['label'] == 'positive').astype(int)
clf = LogisticRegression(max_iter=1000).fit(X, y)
