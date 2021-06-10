from Interpolation import Interpolation
from others import Others
import matplotlib.pyplot as plt
import numpy as np
# 在一个函数下不同方法的3种误差比较
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
def func(x, y, z):
    f = np.cos(x + y + z)
    return f

def run_func(coord):
    node = np.zeros((8, 4))
    for i in range(coord.shape[0]):
        f = func(coord[i][0], coord[i][1], coord[i][2])
        node[i] = np.append(coord[i], [f])
    # P0f = func(cP0[0], cP0[1], cP0[2])
    return node

coord = np.array([[-0.1, -0.1, -0.1], [0.1, -0.1, -0.1], [0.1, 0.1, -0.1], [-0.1, 0.1, -0.1],
                  [-0.1, -0.1, 0.1], [0.1, -0.1, 0.1], [0.1, 0.1, 0.1], [-0.1, 0.1, 0.1]])
coord2 = np.array([[-0.1, 0.1, -0.1], [0.1, 0.1, -0.1], [0.1, 0.3, -0.1], [-0.1, 0.3, -0.1],
                   [-0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.3, 0.1], [-0.1, 0.3, 0.1]])
a = 20
xx1 = np.linspace(-0.1, 0.1, a)
yy1 = np.linspace(0.1, 0.1, 1)
zz1 = np.linspace(-0.1, 0.1, a)
X1, Y1, Z1 = np.meshgrid(xx1, yy1, zz1)
X2, Z2 = np.meshgrid(xx1, zz1)

cP0 = np.empty((1, 3))
for x, y, z in np.nditer((X1, Y1, Z1)):
    cP0 = np.append(cP0, np.array([[x, y, z]]), axis=0)
cP0 = np.delete(cP0, 0, axis=0)
# cP0 = np.array([[0, 0.1, -0.1]])
# f = func(cP0[0][0], cP0[0][1], cP0[0][2])
# print(f)

node = run_func(coord)
ex1 = Interpolation(cP0, node)
P0 = ex1.compute()

node2 = run_func(coord2)
ex2 = Interpolation(cP0, node2)
P02 = ex2.compute()

def rel():
    fig = plt.figure(figsize=(7.5, 6))
    ax2 = fig.add_subplot(3, 2, 1, projection='3d')
    fig.subplots_adjust(left=0.07, bottom=0.13, right=0.89, top=0.94, wspace=0.15, hspace=0.25)

    errorstr = 'rel'
    N = np.vstack((node, node2))
    ptp = np.ptp(N, axis=0)[3]
    delta = np.array([])
    for i in range(cP0.shape[0]):
        f = func(cP0[i][0], cP0[i][1], cP0[i][2])
        ptp = np.ptp(node, axis=0)[3]
        de = abs(round((P0[i][3] - P02[i][3]), 4) / ptp)
        delta = np.append(delta, de)
        # ax2.scatter(cP0[i][1], cP0[i][2], delta, marker='o', Alpha=1)
    delta = delta.reshape(a, a)
    ax2.plot_surface(X2, Z2, delta, cmap='rainbow')
    ax2.set_title('new method', y=-0.3)
    # ax2.grid(False)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_zlabel('ERROR')

    for i in range(5):
        list = ['IDW', 'RBF', 'MQ', 'TPS', 'Poly']
        ax2 = fig.add_subplot(3, 2, i + 2, projection='3d')
        ax2.set_title(list[i], y=-0.3)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        ax2.set_zlabel('ERROR')
        delta = np.array([])
        for j in range(cP0.shape[0]):
            # a = list[i]
            f = func(cP0[j][0], cP0[j][1], cP0[j][2])

            other1 = Others(cP0[j], node)
            other2 = Others(cP0[j], node2)
            P0x1 = other1.compute(method=list[i])
            P0x2 = other2.compute(method=list[i])
            if f == 0 and abs(f - P0[3]) < 1e-10:
                relative_error, absolute_error, ptp_error = 0, 0, 0
            else:
                relative_error = (round((P0x1[3] - P0x2[3]), 4) / f) * 100
                absolute_error = (round((P0x1[3] - P0x2[3]), 4))
                ptp_error = (round((P0x1[3] - P0x2[3]), 4) / ptp) * 100
            errorDict = {'rel': relative_error, 'abs': absolute_error, 'ptp': ptp_error}
            de = errorDict[errorstr]
            delta = np.append(delta, de)

        delta = delta.reshape(a, a)
        ax2.plot_surface(X2, Z2, delta, cmap='rainbow')
        # ax2.scatter(cP0[j][0], cP0[j][1], P0x, marker='o', Alpha=1)

def absoute():
    fig = plt.figure(figsize=(7.5, 6))
    ax2 = fig.add_subplot(3, 2, 1, projection='3d')
    fig.subplots_adjust(left=0.07, bottom=0.13, right=0.89, top=0.94, wspace=0.15, hspace=0.25)

    errorstr = 'abs'
    N = np.vstack((node, node2))
    ptp = np.ptp(N, axis=0)[3]
    delta = np.array([])
    for i in range(cP0.shape[0]):
        f = func(cP0[i][0], cP0[i][1], cP0[i][2])
        ptp = np.ptp(node, axis=0)[3]
        de = abs(round((P0[i][3] - P02[i][3]), 4) / ptp)
        delta = np.append(delta, de)
        # ax2.scatter(cP0[i][1], cP0[i][2], delta, marker='o', Alpha=1)
    delta = delta.reshape(a, a)
    ax2.plot_surface(X2, Z2, delta, cmap='rainbow')
    ax2.set_title('new method', y=-0.3)
    # ax2.grid(False)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_zlabel('ERROR')

    for i in range(5):
        list = ['IDW', 'RBF', 'MQ', 'TPS', 'Poly']
        ax2 = fig.add_subplot(3, 2, i + 2, projection='3d')
        ax2.set_title(list[i], y=-0.3)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        ax2.set_zlabel('ERROR')
        delta = np.array([])
        for j in range(cP0.shape[0]):
            # a = list[i]
            f = func(cP0[j][0], cP0[j][1], cP0[j][2])

            other1 = Others(cP0[j], node)
            other2 = Others(cP0[j], node2)
            P0x1 = other1.compute(method=list[i])
            P0x2 = other2.compute(method=list[i])
            if f == 0 and abs(f - P0[3]) < 1e-10:
                relative_error, absolute_error, ptp_error = 0, 0, 0
            else:
                relative_error = (round((P0x1[3] - P0x2[3]), 4) / f) * 100
                absolute_error = (round((P0x1[3] - P0x2[3]), 4))
                ptp_error = (round((P0x1[3] - P0x2[3]), 4) / ptp) * 100
            errorDict = {'rel': relative_error, 'abs': absolute_error, 'ptp': ptp_error}
            de = errorDict[errorstr]
            delta = np.append(delta, de)

        delta = delta.reshape(a, a)
        ax2.plot_surface(X2, Z2, delta, cmap='rainbow')
        # ax2.scatter(cP0[j][0], cP0[j][1], P0x, marker='o', Alpha=1)

def ptp():
    fig = plt.figure(figsize=(7.5, 6))
    ax2 = fig.add_subplot(3, 2, 1, projection='3d')
    fig.subplots_adjust(left=0.07, bottom=0.13, right=0.89, top=0.94, wspace=0.15, hspace=0.25)

    errorstr = 'ptp'
    N = np.vstack((node, node2))
    ptp = np.ptp(N, axis=0)[3]
    delta = np.array([])
    for i in range(cP0.shape[0]):
        f = func(cP0[i][0], cP0[i][1], cP0[i][2])
        ptp = np.ptp(node, axis=0)[3]
        de = abs(round((P0[i][3] - P02[i][3]), 4) / ptp)
        delta = np.append(delta, de)
        # ax2.scatter(cP0[i][1], cP0[i][2], delta, marker='o', Alpha=1)
    delta = delta.reshape(a, a)
    ax2.plot_surface(X2, Z2, delta, cmap='rainbow')
    ax2.set_title('new method', y=-0.3)
    # ax2.grid(False)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_zlabel('ERROR')

    for i in range(5):
        list = ['IDW', 'RBF', 'MQ', 'TPS', 'Poly']
        ax2 = fig.add_subplot(3, 2, i + 2, projection='3d')
        ax2.set_title(list[i], y=-0.3)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        ax2.set_zlabel('ERROR')
        delta = np.array([])
        for j in range(cP0.shape[0]):
            # a = list[i]
            f = func(cP0[j][0], cP0[j][1], cP0[j][2])

            other1 = Others(cP0[j], node)
            other2 = Others(cP0[j], node2)
            P0x1 = other1.compute(method=list[i])
            P0x2 = other2.compute(method=list[i])
            if f == 0 and abs(f - P0[3]) < 1e-10:
                relative_error, absolute_error, ptp_error = 0, 0, 0
            else:
                relative_error = (round((P0x1[3] - P0x2[3]), 4) / f) * 100
                absolute_error = (round((P0x1[3] - P0x2[3]), 4))
                ptp_error = (round((P0x1[3] - P0x2[3]), 4) / ptp) * 100
            errorDict = {'rel': relative_error, 'abs': absolute_error, 'ptp': ptp_error}
            de = errorDict[errorstr]
            delta = np.append(delta, de)

        delta = delta.reshape(a, a)
        ax2.plot_surface(X2, Z2, delta, cmap='rainbow')
        # ax2.scatter(cP0[j][0], cP0[j][1], P0x, marker='o', Alpha=1)

rel()
absoute()
ptp()

plt.show()
