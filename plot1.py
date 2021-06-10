import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure()  # 定义新的三维坐标轴
ax1 = plt.axes(projection='3d')
#ax2 = Axes3D(fig)

# 定义三维数据
xx1 = np.arange(-1, 2, 1)
yy1 = np.arange(-1, 2, 1)
zz1 = np.arange(-1, 2, 1)
X1, Y1, Z1 = np.meshgrid(xx1, yy1, zz1)
#Z1 = np.arctan(X1*Y1)

# 作图
#ax1.plot_surface(X1, Y1, Z1, rstride = 1, cstride = 1, cmap='rainbow', Alpha=0.5)

#ax3.contour(X,Y,Z,offset=-2, cmap = 'rainbow')#绘制等高线
#ax3.text(0, 0, 0, "red", color='red')
#plt.plot(X1,Y1)
# for xs,ys in [xx1,yy1]:
#     ax1.scatter(xs, ys, zs=0, marker='o', Alpha=1)
ax1.scatter(X1, Y1, Z1, marker='o',Alpha=0.8)

#ax1.scatter(X1, Y1, zs=1, marker='o', cmap='rainbow',Alpha=0.8)

ax1.set_xlabel('X Label')
ax1.set_ylabel('Y Label')
ax1.set_zlabel('Z Label')
plt.show()

