import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import LinearLocator

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def func(x, y, z):
    f = np.cos(x + y + z)
    return f


# 定义三维数据
xx1 = np.arange(-0.1, 0.1, 0.01)
yy1 = np.arange(-0.1, 0.1, 0.01)
zz1 = np.arange(-0.1, 0.1, 0.01)
X1, Y1 = np.meshgrid(xx1, yy1)
F1 = func(X1, Y1, 0.05)
X2, Z2 = np.meshgrid(xx1, zz1)
F2 = func(X2, 0.09, Z2)
Y3, Z3 = np.meshgrid(yy1, zz1)
F3 = func(0, Y3, Z3)

# 作图
# ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow', Alpha=0.5)


fig = plt.figure(figsize=(12, 3.5))  # 定义新的三维坐标轴
# ax1 = plt.axes(projection='3d')
# fig.subplots_adjust(left=0.02, bottom=0.13, right=0.96, top=0.94, wspace=0.03, hspace=0.25)

norm = Normalize(vmin=0.96, vmax=1.001)

ax1 = fig.add_subplot(1, 3, 1, projection='3d')
cset1 = ax1.contourf(F3, Y3, Z3, zdir='x', offset=0, cmap='rainbow', alpha=0.8, norm=norm)
ax1.scatter(0, 0.09, 0.05, s=50, marker='^', c='b', label='待插值点')
for i in range(12):
    coord = np.array([[[0.1, -0.1], [-0.1, -0.1], [-0.1, -0.1]], [[0.1, -0.1], [0.1, 0.1], [-0.1, -0.1]],
                      [[0.1, -0.1], [-0.1, -0.1], [0.1, 0.1]], [[0.1, -0.1], [0.1, 0.1], [0.1, 0.1]],
                      [[0.1, 0.1], [-0.1, 0.1], [-0.1, -0.1]], [[0.1, 0.1], [-0.1, 0.1], [0.1, 0.1]],
                      [[-0.1, -0.1], [-0.1, 0.1], [-0.1, -0.1]], [[-0.1, -0.1], [-0.1, 0.1], [0.1, 0.1]],
                      [[0.1, 0.1], [-0.1, -0.1], [-0.1, 0.1]], [[0.1, 0.1], [0.1, 0.1], [-0.1, 0.1]],
                      [[-0.1, -0.1], [-0.1, -0.1], [-0.1, 0.1]], [[-0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]]])
    line = ax1.plot(coord[i][0], coord[i][1], coord[i][2], c='black')
xg = np.linspace(-0.1, 0.1, 2)
yg, zg = xg, xg
xs, ys, zs = np.meshgrid(xg, yg, zg)
ax1.scatter(xs, ys, zs, s=50, marker='o', Alpha=1, c='black', label='网格节点')
# fig.colorbar(cset1, ax=ax1, shrink=0.7, aspect=10)
ax1.set_xlim(-0.1, 0.1)
ax1.set_ylim(-0.1, 0.1)
ax1.set_zlim(-0.1, 0.1)
ax1.zaxis.set_major_locator(LinearLocator(5))
ax1.xaxis.set_major_locator(LinearLocator(5))
ax1.yaxis.set_major_locator(LinearLocator(5))
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('x=0.00', y=-0.15)

ax1 = fig.add_subplot(1, 3, 2, projection='3d')
cset2 = ax1.contourf(X2, F2, Z2, zdir='y', offset=0.09, cmap='rainbow', alpha=0.8, norm=norm)
ax1.scatter(0, 0.09, 0.05, s=50, marker='^', c='b', label='待插值点')
for i in range(12):
    coord = np.array([[[0.1, -0.1], [-0.1, -0.1], [-0.1, -0.1]], [[0.1, -0.1], [0.1, 0.1], [-0.1, -0.1]],
                      [[0.1, -0.1], [-0.1, -0.1], [0.1, 0.1]], [[0.1, -0.1], [0.1, 0.1], [0.1, 0.1]],
                      [[0.1, 0.1], [-0.1, 0.1], [-0.1, -0.1]], [[0.1, 0.1], [-0.1, 0.1], [0.1, 0.1]],
                      [[-0.1, -0.1], [-0.1, 0.1], [-0.1, -0.1]], [[-0.1, -0.1], [-0.1, 0.1], [0.1, 0.1]],
                      [[0.1, 0.1], [-0.1, -0.1], [-0.1, 0.1]], [[0.1, 0.1], [0.1, 0.1], [-0.1, 0.1]],
                      [[-0.1, -0.1], [-0.1, -0.1], [-0.1, 0.1]], [[-0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]]])
    line = ax1.plot(coord[i][0], coord[i][1], coord[i][2], c='black')
xg = np.linspace(-0.1, 0.1, 2)
yg, zg = xg, xg
xs, ys, zs = np.meshgrid(xg, yg, zg)
ax1.scatter(xs, ys, zs, s=50, marker='o', Alpha=1, c='black', label='网格节点')
# fig.colorbar(cset2, ax=ax1, shrink=0.7, aspect=10)
ax1.set_xlim(-0.1, 0.1)
ax1.set_ylim(-0.1, 0.1)
ax1.set_zlim(-0.1, 0.1)
ax1.zaxis.set_major_locator(LinearLocator(5))
ax1.xaxis.set_major_locator(LinearLocator(5))
ax1.yaxis.set_major_locator(LinearLocator(5))
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('y=0.09', y=-0.15)

ax1 = fig.add_subplot(1, 3, 3, projection='3d')
cset3 = ax1.contourf(X1, Y1, F1, zdir='z', offset=0.05, cmap='rainbow', alpha=0.8, norm=norm)
ax1.scatter(0, 0.09, 0.05, s=50, marker='^', c='b', label='待插值点')
for i in range(12):
    coord = np.array([[[0.1, -0.1], [-0.1, -0.1], [-0.1, -0.1]], [[0.1, -0.1], [0.1, 0.1], [-0.1, -0.1]],
                      [[0.1, -0.1], [-0.1, -0.1], [0.1, 0.1]], [[0.1, -0.1], [0.1, 0.1], [0.1, 0.1]],
                      [[0.1, 0.1], [-0.1, 0.1], [-0.1, -0.1]], [[0.1, 0.1], [-0.1, 0.1], [0.1, 0.1]],
                      [[-0.1, -0.1], [-0.1, 0.1], [-0.1, -0.1]], [[-0.1, -0.1], [-0.1, 0.1], [0.1, 0.1]],
                      [[0.1, 0.1], [-0.1, -0.1], [-0.1, 0.1]], [[0.1, 0.1], [0.1, 0.1], [-0.1, 0.1]],
                      [[-0.1, -0.1], [-0.1, -0.1], [-0.1, 0.1]], [[-0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]]])
    line = ax1.plot(coord[i][0], coord[i][1], coord[i][2], c='black')
xg = np.linspace(-0.1, 0.1, 2)
yg, zg = xg, xg
xs, ys, zs = np.meshgrid(xg, yg, zg)
ax1.scatter(xs, ys, zs, s=50, marker='o', Alpha=1, c='black', label='网格节点')
# fig.colorbar(cset3, ax=ax1, shrink=0.7, aspect=10)
# fig.subplots_adjust(right=0.9)
fig.subplots_adjust(left=0.02, bottom=0.13, right=0.9, top=0.94, wspace=0.2, hspace=0.25)

# colorbar 左 下 宽 高
l = 0.92
b = 0.15
w = 0.015
h = 0.7

# 对应 l,b,w,h；设置colorbar位置；
rect = [l, b, w, h]
cbar_ax = fig.add_axes(rect)
cb = plt.colorbar(cset2, cax=cbar_ax)

# 设置colorbar标签字体等
cb.ax.tick_params(labelsize=10)  # 设置色标刻度字体大小。

ax1.set_xlim(-0.1, 0.1)
ax1.set_ylim(-0.1, 0.1)
ax1.set_zlim(-0.1, 0.1)
ax1.zaxis.set_major_locator(LinearLocator(5))
ax1.xaxis.set_major_locator(LinearLocator(5))
ax1.yaxis.set_major_locator(LinearLocator(5))
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('z=0.05', y=-0.15)

ax1.legend()
plt.show()
