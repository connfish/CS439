# value counts / contingency
import pandas as pd
import numpy as np
df = pd.read_csv('data')
df['col'].value_counts()
pd.crosstab(df['r'], df['c'], margins=True, normalize='index')

# groupby summary
(df.groupby('section')[['calc', 'prob']]
   .agg(['mean','median','std']).round(2))

# pivot
df.pivot_table(index='role', columns='section',
               values='total_score', aggfunc='mean', fill_value=0)

# scaling (z-score)
cols = ['calc','prob','viz']
df[cols] = (df[cols] - df[cols].mean())/df[cols].std()
