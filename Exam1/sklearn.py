from sklearn.linear_model import LinearRegression
import matplotlib as plt
import pandas as pd
import statsmodels.api as sm
df = pd.read_csv('data')
X = df[['x1','x2']]; y = df['y']
lr = LinearRegression().fit(X, y)
resid = y - lr.predict(X)
sm.qqplot(resid, line='45'); plt.show()
plt.scatter(lr.predict(X), resid); plt.axhline(0); plt.show()
