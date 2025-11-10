import numpy as np, matplotlib.pyplot as plt
x = np.linspace(0.01, 10, 500)
plt.plot(x, np.sin(x), label='sin x')
plt.plot(x, x**3, label='x^3')
plt.plot(x, np.log2(x), label='log2 x')
plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.show()
