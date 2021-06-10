from Interpolation import Interpolation
from others import Others
import matplotlib.pyplot as plt
import numpy as np
# 同一方法在不同函数下ptp误差比较
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

coord = np.array([[-0.1, -0.1, -0.1], [0.1, -0.1, -0.1], [0.1, 0.1, -0.1], [-0.1, 0.1, -0.1],
                  [-0.1, -0.1, 0.1], [0.1, -0.1, 0.1], [0.1, 0.1, 0.1], [-0.1, 0.1, 0.1]])
coord2 = np.array([[-0.1, 0.1, -0.1], [0.1, 0.1, -0.1], [0.1, 0.3, -0.1], [-0.1, 0.3, -0.1],
                   [-0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.3, 0.1], [-0.1, 0.3, 0.1]])

a = 10
xx1 = np.linspace(-0.1, 0.1, a)
yy1 = np.linspace(0.1, 0.1, 1)
zz1 = np.linspace(-0.1, 0.1, a)
X1, Y1, Z1 = np.meshgrid(xx1, yy1, zz1)
X2, Z2 = np.meshgrid(xx1, zz1)

cP0 = np.empty((1, 3))
for x, y, z in np.nditer((X1, Y1, Z1)):
    cP0 = np.append(cP0, np.array([[x, y, z]]), axis=0)
cP0 = np.delete(cP0, 0, axis=0)


def sinFunc(x, y, z):
    f = np.sin(x + y + z)
    return f


def cosFunc(x, y, z):
    f = np.cos(x + y + z)
    return f


def arctanFunc(x, y, z):
    f = -np.arctan(x + y + z)
    return f


funcDict = {'sin': sinFunc, 'cos': cosFunc, 'arctan': arctanFunc}
funcList = ['sin', 'cos', 'arctan']
methodList = ['newMethod', 'IDW', 'RBF', 'MQ', 'TPC', 'Poly']


def run_func(coord, whichfunc):
    node = np.zeros((8, 4))
    for i in range(coord.shape[0]):
        f = funcDict[whichfunc](coord[i][0], coord[i][1], coord[i][2])
        node[i] = np.append(coord[i], [f])
    return node


def printPlot(cP0):
    for i in range(3):
        node = run_func(coord, funcList[i])
        node2 = run_func(coord2, funcList[i])
        N = np.vstack((node, node2))
        ptp = np.ptp(N, axis=0)[3]
        for j in range(6):
            delta = np.array([])
            ax1 = fig.add_subplot(3, 6, i * 6 + j + 1, projection='3d')
            for k in range(cP0.shape[0]):
                if methodList[j] != 'newMethod':
                    other1 = Others(cP0[k], node)
                    other2 = Others(cP0[k], node2)
                    P0x1 = other1.compute(method=methodList[j])
                    P0x2 = other2.compute(method=methodList[j])
                    de = abs(round((P0x1[3] - P0x2[3]), 4) / ptp) * 100
                    delta = np.append(delta, de)
                    print(666)
                else:
                    ex1 = Interpolation(cP0, node)
                    P0 = ex1.compute()
                    ex2 = Interpolation(cP0, node2)
                    P02 = ex2.compute()
                    de = abs(round((P0[i][3] - P02[i][3]), 4) / ptp)
                    delta = np.append(delta, de)
            delta = delta.reshape(int(np.sqrt(delta.size)), int(np.sqrt(delta.size)))
            ax1.plot_surface(X2, Z2, delta, cmap='rainbow')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('DELTA')
            if i == 2:
                ax1.set_title(methodList[j], y=-0.3)


fig = plt.figure(figsize=(14, 8))
fig.subplots_adjust(left=0.05, bottom=0.11, right=0.97, top=0.96, wspace=0.16, hspace=0.25)
printPlot(cP0)

plt.show()
