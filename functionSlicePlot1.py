import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.colors import Normalize

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def func(x, y, z):
    f = np.cos(x + y + z)
    return f
# 定义三维数据
xx1 = np.arange(-0.1, 0.1, 0.01)
yy1 = np.arange(-0.1, 0.1, 0.01)
zz1 = np.arange(-0.1, 0.1, 0.01)
X1, Z1 = np.meshgrid(xx1, zz1)
F1 = np.sin(X1+0.1+Z1)
F2 = np.cos(X1+ 0.1+ Z1)
F3 = -np.arctan(X1+0.1+Z1)

# 作图
# ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow', Alpha=0.5)


fig = plt.figure(figsize = (14.5,4.5))  # 定义新的三维坐标轴
# ax1 = plt.axes(projection='3d')
# fig.subplots_adjust(left=0.02, bottom=0.13, right=0.96, top=0.94, wspace=0.03, hspace=0.25)

norm = Normalize(vmin=0.96, vmax=1.001)

ax1 = fig.add_subplot(1, 3, 1, projection='3d')
cset1 = ax1.contourf(X1, F1, Z1, zdir='y', offset=0.1, cmap='rainbow', alpha=0.8)
ax1.scatter(0.1,0.1,0, s=50, marker='^',c='b',label='待插值点')
for i in range(12):
    coord = np.array([[[0.1, -0.1], [-0.1, -0.1], [-0.1, -0.1]],[[0.1, -0.1], [0.1, 0.1], [-0.1, -0.1]],
                      [[0.1, -0.1], [-0.1, -0.1], [0.1, 0.1]],[[0.1, -0.1], [0.1, 0.1], [0.1, 0.1]],
                      [[0.1, 0.1], [-0.1, 0.1], [-0.1, -0.1]],[[0.1, 0.1], [-0.1, 0.1], [0.1, 0.1]],
                      [[-0.1, -0.1], [-0.1, 0.1], [-0.1, -0.1]],[[-0.1, -0.1], [-0.1, 0.1], [0.1, 0.1]],
                      [[0.1, 0.1], [-0.1, -0.1], [-0.1, 0.1]],[[0.1, 0.1], [0.1, 0.1], [-0.1, 0.1]],
                      [[-0.1, -0.1], [-0.1, -0.1], [-0.1, 0.1]],[[-0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]]])
    line = ax1.plot(coord[i][0], coord[i][1],coord[i][2],c='black')
xg = np.linspace(-0.1,0.1,2)
yg, zg = xg, xg
xs, ys, zs = np.meshgrid(xg,yg,zg)
ax1.scatter(xs, ys, zs, s=50, marker='o', Alpha=1,c='black',label='网格节点')
fig.colorbar(cset1, ax=ax1, shrink=0.7, aspect=10)
ax1.set_xlim(-0.1, 0.1)
ax1.set_ylim(-0.1, 0.1)
ax1.set_zlim(-0.1, 0.1)
ax1.zaxis.set_major_locator(LinearLocator(5))
ax1.xaxis.set_major_locator(LinearLocator(5))
ax1.yaxis.set_major_locator(LinearLocator(5))
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend()
# ax1.set_title('y=0.09', y=-0.15)

ax1 = fig.add_subplot(1, 3, 2, projection='3d')
cset2 = ax1.contourf(X1, F2, Z1, zdir='y', offset=0.1, cmap='rainbow', alpha=0.8)
ax1.scatter(0.1,0.1,0, s=50, marker='^',c='b',label='待插值点')
for i in range(12):
    coord = np.array([[[0.1, -0.1], [-0.1, -0.1], [-0.1, -0.1]],[[0.1, -0.1], [0.1, 0.1], [-0.1, -0.1]],
                      [[0.1, -0.1], [-0.1, -0.1], [0.1, 0.1]],[[0.1, -0.1], [0.1, 0.1], [0.1, 0.1]],
                      [[0.1, 0.1], [-0.1, 0.1], [-0.1, -0.1]],[[0.1, 0.1], [-0.1, 0.1], [0.1, 0.1]],
                      [[-0.1, -0.1], [-0.1, 0.1], [-0.1, -0.1]],[[-0.1, -0.1], [-0.1, 0.1], [0.1, 0.1]],
                      [[0.1, 0.1], [-0.1, -0.1], [-0.1, 0.1]],[[0.1, 0.1], [0.1, 0.1], [-0.1, 0.1]],
                      [[-0.1, -0.1], [-0.1, -0.1], [-0.1, 0.1]],[[-0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]]])
    line = ax1.plot(coord[i][0], coord[i][1],coord[i][2],c='black')
xg = np.linspace(-0.1,0.1,2)
yg, zg = xg, xg
xs, ys, zs = np.meshgrid(xg,yg,zg)
ax1.scatter(xs, ys, zs, s=50, marker='o', Alpha=1,c='black',label='网格节点')
fig.colorbar(cset2, ax=ax1, shrink=0.7, aspect=10)
ax1.set_xlim(-0.1, 0.1)
ax1.set_ylim(-0.1, 0.1)
ax1.set_zlim(-0.1, 0.1)
ax1.zaxis.set_major_locator(LinearLocator(5))
ax1.xaxis.set_major_locator(LinearLocator(5))
ax1.yaxis.set_major_locator(LinearLocator(5))
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend()

ax1 = fig.add_subplot(1, 3, 3, projection='3d')
cset3 = ax1.contourf(X1, F3, Z1, zdir='y', offset=0.1, cmap='rainbow', alpha=0.8)
ax1.scatter(0.1,0.1,0, s=50, marker='^',c='b',label='待插值点')
for i in range(12):
    coord = np.array([[[0.1, -0.1], [-0.1, -0.1], [-0.1, -0.1]],[[0.1, -0.1], [0.1, 0.1], [-0.1, -0.1]],
                      [[0.1, -0.1], [-0.1, -0.1], [0.1, 0.1]],[[0.1, -0.1], [0.1, 0.1], [0.1, 0.1]],
                      [[0.1, 0.1], [-0.1, 0.1], [-0.1, -0.1]],[[0.1, 0.1], [-0.1, 0.1], [0.1, 0.1]],
                      [[-0.1, -0.1], [-0.1, 0.1], [-0.1, -0.1]],[[-0.1, -0.1], [-0.1, 0.1], [0.1, 0.1]],
                      [[0.1, 0.1], [-0.1, -0.1], [-0.1, 0.1]],[[0.1, 0.1], [0.1, 0.1], [-0.1, 0.1]],
                      [[-0.1, -0.1], [-0.1, -0.1], [-0.1, 0.1]],[[-0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]]])
    line = ax1.plot(coord[i][0], coord[i][1],coord[i][2],c='black')
xg = np.linspace(-0.1,0.1,2)
yg, zg = xg, xg
xs, ys, zs = np.meshgrid(xg,yg,zg)
ax1.scatter(xs, ys, zs, s=50, marker='o', Alpha=1,c='black',label='网格节点')
fig.colorbar(cset3, ax=ax1, shrink=0.7, aspect=10)
ax1.set_xlim(-0.1, 0.1)
ax1.set_ylim(-0.1, 0.1)
ax1.set_zlim(-0.1, 0.1)
ax1.zaxis.set_major_locator(LinearLocator(5))
ax1.xaxis.set_major_locator(LinearLocator(5))
ax1.yaxis.set_major_locator(LinearLocator(5))
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
# ax1.set_title('y=0.09', y=-0.15)
# fig.colorbar(cset3, ax=ax1, shrink=0.7, aspect=10)

fig.subplots_adjust(left=0.02, bottom=0.13, right=0.96, top=0.94, wspace=0.07, hspace=0.25)

ax1.legend()
plt.show()
