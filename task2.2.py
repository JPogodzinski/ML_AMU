import numpy as np
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(10,8))

x = np.linspace(-10, 10, 100)
a=6
b=2
c=2
y=(a-4)*x**2+(b-5)*x+(c-6)

g=np.exp(x)/(np.exp(x)+1)


ax=fig.add_subplot(2,1,1)
ax.set_ylabel("os y")
ax.set_xlabel("os x")

line1=ax.plot(x,y, color='blue', lw=2)
line2=ax.plot(x,g, color='red', lw=2)

plt.show()