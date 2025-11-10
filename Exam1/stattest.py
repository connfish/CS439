from scipy import stats
import numpy as np
# 1-prop z-test (manual)
p0=0.5; x=15; n=20
phat = x/n
z = (phat - p0)/np.sqrt(p0*(1-p0)/n)
p = 1 - stats.norm.cdf(z)  # one-sided

# 2-sample t-test (Welch)
#stats.ttest_ind(groupA, groupB, equal_var=False)
