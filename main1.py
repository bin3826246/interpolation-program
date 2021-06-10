from Interpolation import Interpolation
from others import Others
import matplotlib.pyplot as plt
import numpy as np
# 同一插值方法不同函数比较
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def sinFunc(x, y, z):
    f = np.sin(x + y + z)
    return f


def cosFunc(x, y, z):
    f = np.cos(x + y + z)
    return f


def arctanFunc(x, y, z):
    f = -np.arctan(x + y + z)
    return f


coord = np.array([[-0.1, -0.1, -0.1], [0.1, -0.1, -0.1], [0.1, 0.1, -0.1], [-0.1, 0.1, -0.1],
                  [-0.1, -0.1, 0.1], [0.1, -0.1, 0.1], [0.1, 0.1, 0.1], [-0.1, 0.1, 0.1]])

a = 10
xx1 = np.linspace(-0.1, 0.1, a)
yy1 = np.linspace(-0.1, 0.1, a)
zz1 = np.linspace(0.03, 0.03, 1)
X1, Y1, Z1 = np.meshgrid(xx1, yy1, zz1)
X2, Y2 = np.meshgrid(xx1, yy1)

cP0 = np.empty((1, 3))
for x, y, z in np.nditer((X1, Y1, Z1)):
    cP0 = np.append(cP0, np.array([[x, y, z]]), axis=0)
cP0 = np.delete(cP0, 0, axis=0)

fig = plt.figure(figsize=(14, 8))
fig.subplots_adjust(left=0.05, bottom=0.11, right=0.97, top=0.96, wspace=0.16, hspace=0.25)


def computeError(f, result, node, whicherror):
    if f == 0 and abs(f - result) < 1e-10:
        relative_error = 0
    else:
        relative_error = abs((f - result) / f) * 100
    absolute_error = f - result
    ptp = np.ptp(node, axis=0)[3]
    ptp_error = ((f - result) / ptp) * 100
    errorDict = {'rel': relative_error, 'abs': absolute_error, 'ptp': ptp_error}
    error = errorDict[whicherror]
    return error


methodList = ['newMethod', 'IDW', 'RBF', 'MQ', 'TPC', 'Poly']
funcDict = {'sin': sinFunc, 'cos': cosFunc, 'arctan': arctanFunc}
funcList = ['sin', 'cos', 'arctan']


def run_func(coord, whichfunc):
    node = np.zeros((8, 4))
    for i in range(coord.shape[0]):
        f = funcDict[whichfunc](coord[i][0], coord[i][1], coord[i][2])
        node[i] = np.append(coord[i], [f])

    return node


def printPlot(cP0, whicherror):
    for i in range(3):
        node = run_func(coord, funcList[i])
        for j in range(6):
            error = np.array([])
            ax1 = fig.add_subplot(3, 6, i * 6 + j + 1, projection='3d')
            for k in range(cP0.shape[0]):
                f = funcDict[funcList[i]](cP0[k][0], cP0[k][1], cP0[k][2])
                if methodList[j] != 'newMethod':
                    ptp = np.ptp(node, axis=0)[3]
                    other = Others(cP0[k], node)
                    error = np.append(error, other.compute_error(f, ptp, method=methodList[j], whicherror=whicherror))
                    print(666)
                else:
                    ex1 = Interpolation(cP0, node)
                    P0 = ex1.compute()
                    error = np.append(error, computeError(f, P0[k][3], node, whicherror))
            error = error.reshape(int(np.sqrt(error.size)), int(np.sqrt(error.size)))
            ax1.plot_surface(X2, Y2, error, cmap='rainbow')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('ERROR')
            if i == 2:
                ax1.set_title(methodList[j], y=-0.3)


printPlot(cP0, 'ptp')

plt.show()
