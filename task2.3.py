from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

x = np.arange(-10, 10, 0.25)
y=np.arange(-10,10,0.25)
x, y = np.meshgrid(x, y)
z=-(x**2+y**3)

surf=ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet,
        linewidth=0, antialiased=True)

plt.show()