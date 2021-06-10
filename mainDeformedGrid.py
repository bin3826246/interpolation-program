from Interpolation import Interpolation
from others import Others
import matplotlib.pyplot as plt
import numpy as np
import sys

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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

coord = np.array([[-0.1, -0.1, 0], [0.1, -0.1, -0.1], [0.1, 0.1, -0.1], [-0.1, 0.1, 0],
                  [-0.1, -0.1, 0.2], [0.1, -0.1, 0.1], [0.1, 0.1, 0.1], [-0.1, 0.1, 0.2]])

# a = 10
# xx1 = np.linspace(0, 0, 1)
# yy1 = np.linspace(-0.1, 0.1, a)
# zz1 = np.linspace(-0.1, 0.1, a)
# X1, Y1, Z1 = np.meshgrid(xx1, yy1, zz1)
# Y2, Z2 = np.meshgrid(yy1, zz1)
#
# cP0 = np.empty((1, 3))
# for x, y, z in np.nditer((X1, Y1, Z1)):
#     cP0 = np.append(cP0, np.array([[x, y, z]]), axis=0)
# cP0 = np.delete(cP0, 0, axis=0)
# # cP0 = np.array([[0, 0.09, 0.05]])
#
# node = run_func(coord)
# ex1 = Interpolation(cP0, node)
# P0 = ex1.compute()
#
# fig = plt.figure(figsize=(7.5, 6))
# ax1 = fig.add_subplot(3, 2, 1, projection='3d')
# fig.subplots_adjust(left=0.07, bottom=0.13, right=0.89, top=0.94, wspace=0.15, hspace=0.25)
#
# errorstr = 'ptp'
# error = np.array([])
# for i in range(cP0.shape[0]):
#     f = func(cP0[i][0], cP0[i][1], cP0[i][2])
#     if f == 0 and abs(f - P0[i][3]) < 1e-2:
#         relative_error = 0
#     else:
#         relative_error = abs((f - P0[i][3]) / f)*100
#     absolute_error = f - P0[i][3]
#     ptp = np.ptp(node, axis=0)[3]
#     ptp_error = ((f - P0[i][3]) / ptp)*100
#     errorDict = {'rel': relative_error, 'abs': absolute_error, 'ptp': ptp_error}
#     error = np.append(error, errorDict[errorstr])
# error = error.reshape(int(np.sqrt(error.size)), int(np.sqrt(error.size)))
# ax1.plot_surface(Y2, Z2, error, cmap='rainbow', rstride=1, cstride=1)
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# ax1.set_zlabel('ERROR')
# ax1.set_title('new method', y=-0.3)
#
# for i in range(5):
#     list = ['IDW', 'RBF', 'MQ', 'TPC', 'Poly']
#     ax1 = fig.add_subplot(3, 2, i + 2, projection='3d')
#     ax1.set_title(list[i], y=-0.3)
#     errorx = np.array([])
#     for j in range(cP0.shape[0]):
#         # a = list[i]
#         f = func(cP0[j][0], cP0[j][1], cP0[j][2])
#         ptp = np.ptp(node, axis=0)[3]
#         other = Others(cP0[j], node)
#         errorx = np.append(errorx, other.compute_error(f, ptp, method=list[i], whicherror=errorstr))
#         # print(list[i], errorx)
#         # ax1.scatter(cP0[j][0], cP0[j][1], P0x, marker='o', Alpha=1)
#     errorx = errorx.reshape(int(np.sqrt(errorx.size)), int(np.sqrt(errorx.size)))
#     ax1.plot_surface(Y2, Z2, errorx, cmap='rainbow', rstride=1, cstride=1)
#     ax1.set_xlabel('X')
#     ax1.set_ylabel('Y')
#     ax1.set_zlabel('ERROR')
#     ax1.set_title(list[i], y=-0.3)
def xsurface():
    a = 10
    xx1 = np.linspace(0, 0, 1)
    yy1 = np.linspace(-0.1, 0.1, a)
    zz1 = np.linspace(-0.05, 0.15, a)
    XX, YY, ZZ = np.meshgrid(xx1, yy1, zz1)
    Y2, Z2 = np.meshgrid(yy1, zz1)

    cP0 = np.empty((1, 3))
    for x, y, z in np.nditer((XX, YY, ZZ)):
        cP0 = np.append(cP0, np.array([[x, y, z]]), axis=0)
    cP0 = np.delete(cP0, 0, axis=0)
    # cP0 = np.array([[0.1, 0.1, 0]])
    # f = func(cP0[0][0], cP0[0][1], cP0[0][2])
    # print(f)

    node = run_func(coord)
    ex1 = Interpolation(cP0, node)
    P0 = ex1.compute()

    fig = plt.figure(figsize=(7.5, 6))
    ax1 = fig.add_subplot(3, 2, 1, projection='3d')
    fig.subplots_adjust(left=0.07, bottom=0.13, right=0.89, top=0.94, wspace=0.15, hspace=0.25)

    errorstr = 'ptp'
    error = np.array([])
    for i in range(cP0.shape[0]):
        f = func(cP0[i][0], cP0[i][1], cP0[i][2])
        if f == 0 and abs(f - P0[i][3]) < 1e-1:
            relative_error = 0
        else:
            relative_error = abs((f - P0[i][3]) / f)*100
        absolute_error = f - P0[i][3]
        ptp = np.ptp(node, axis=0)[3]
        ptp_error = ((f - P0[i][3]) / ptp)*100
        errorDict = {'rel': relative_error, 'abs': absolute_error, 'ptp': ptp_error}
        error = np.append(error, errorDict[errorstr])
        # ax1.scatter(cP0[i][0], cP0[i][1], ptp_error, marker='o', Alpha=1)
    error = error.reshape(int(np.sqrt(error.size)), int(np.sqrt(error.size)))
    ax1.plot_surface(Y2, Z2, error, cmap='rainbow')
    ax1.set_xlabel('Y')
    ax1.set_ylabel('Z')
    ax1.set_zlabel('ERROR')
    ax1.set_title('new method', y=-0.3)

    for i in range(5):
        list = ['IDW', 'RBF', 'MQ', 'TPS', 'Poly']
        ax1 = fig.add_subplot(3, 2, i + 2, projection='3d')
        ax1.set_title(list[i], y=-0.3)
        errorx = np.array([])
        for j in range(cP0.shape[0]):
            f = func(cP0[j][0], cP0[j][1], cP0[j][2])
            ptp = np.ptp(node, axis=0)[3]
            other = Others(cP0[j], node)
            errorx = np.append(errorx, other.compute_error(f, ptp, method=list[i], whicherror=errorstr))
            # ax1.scatter(cP0[j][0], cP0[j][1], P0x, marker='o', Alpha=1)
        errorx = errorx.reshape(int(np.sqrt(errorx.size)), int(np.sqrt(errorx.size)))
        ax1.plot_surface(Y2, Z2, errorx, cmap='rainbow')
        ax1.set_xlabel('Y')
        ax1.set_ylabel('Z')
        ax1.set_zlabel('ERROR')
        ax1.set_title(list[i], y=-0.3)

def ysurface():
    a = 10
    xx1 = np.linspace(-0.1, 0.1, a)
    yy1 = np.linspace(0.09, 0.09, 1)
    zz1 = np.linspace(-0.1, 0.2, a)
    XX, YY, ZZ = np.meshgrid(xx1, yy1, zz1)
    X3, Z3 = np.meshgrid(xx1, zz1)

    cP0 = np.empty((1, 3))
    for x, y, z in np.nditer((XX, YY, ZZ)):
        cP0 = np.append(cP0, np.array([[x, y, z]]), axis=0)
    cP0 = np.delete(cP0, 0, axis=0)
    # cP0 = np.array([[0.1, 0.1, 0]])
    # f = func(cP0[0][0], cP0[0][1], cP0[0][2])
    # print(f)

    node = run_func(coord)
    ex1 = Interpolation(cP0, node)
    P0 = ex1.compute()

    fig = plt.figure(figsize=(7.5, 6))
    ax1 = fig.add_subplot(3, 2, 1, projection='3d')
    fig.subplots_adjust(left=0.07, bottom=0.13, right=0.89, top=0.94, wspace=0.15, hspace=0.25)

    errorstr = 'ptp'
    error = np.array([])
    for i in range(cP0.shape[0]):
        f = func(cP0[i][0], cP0[i][1], cP0[i][2])
        if f == 0 and abs(f - P0[i][3]) < 1e-10:
            relative_error = 0
        else:
            relative_error = abs((f - P0[i][3]) / f) * 100
        absolute_error = f - P0[i][3]
        ptp = np.ptp(node, axis=0)[3]
        ptp_error = ((f - P0[i][3]) / ptp) * 100
        errorDict = {'rel': relative_error, 'abs': absolute_error, 'ptp': ptp_error}
        if cP0[i][0]+2*cP0[i][2]>0.3 or cP0[i][0]+2*cP0[i][2]<-0.1:
            error = np.append(error, False)
        else:
            error = np.append(error, errorDict[errorstr])
    error = error.reshape(int(np.sqrt(error.size)), int(np.sqrt(error.size)))
    ax1.plot_surface(X3, Z3, error, cmap='rainbow')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.set_zlabel('ERROR')
    ax1.set_title('new method', y=-0.3)

    for i in range(5):
        list = ['IDW', 'RBF', 'MQ', 'TPS', 'Poly']
        ax1 = fig.add_subplot(3, 2, i + 2, projection='3d')
        ax1.set_title(list[i], y=-0.3)
        errorx = np.array([])
        for j in range(cP0.shape[0]):
            f = func(cP0[j][0], cP0[j][1], cP0[j][2])
            ptp = np.ptp(node, axis=0)[3]
            other = Others(cP0[j], node)
            if cP0[j][0] + 2 * cP0[j][2] > 0.3 or cP0[j][0] + 2 * cP0[j][2] < -0.1:
                errorx = np.append(errorx, False)
            else:
                errorx = np.append(errorx, other.compute_error(f, ptp, method=list[i], whicherror=errorstr))
            # ax1.scatter(cP0[j][0], cP0[j][1], P0x, marker='o', Alpha=1)
        errorx = errorx.reshape(int(np.sqrt(errorx.size)), int(np.sqrt(errorx.size)))
        ax1.plot_surface(X3, Z3, errorx, cmap='rainbow')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Z')
        ax1.set_zlabel('ERROR')
        ax1.set_title(list[i], y=-0.3)

def zsurface():
    a = 10
    xx1 = np.linspace(-0.1, 0.1, a)
    yy1 = np.linspace(-0.1, 0.1, a)
    zz1 = np.linspace(0.05, 0.05, 1)
    XX, YY, ZZ = np.meshgrid(xx1, yy1, zz1)
    X1, Y1 = np.meshgrid(xx1, yy1)

    cP0 = np.empty((1, 3))
    for x, y, z in np.nditer((XX, YY, ZZ)):
        cP0 = np.append(cP0, np.array([[x, y, z]]), axis=0)
    cP0 = np.delete(cP0, 0, axis=0)
    # cP0 = np.array([[0.1, 0.1, 0]])
    # f = func(cP0[0][0], cP0[0][1], cP0[0][2])
    # print(f)

    node = run_func(coord)
    ex1 = Interpolation(cP0, node)
    P0 = ex1.compute()

    fig = plt.figure(figsize=(7.5, 6))
    ax1 = fig.add_subplot(3, 2, 1, projection='3d')
    fig.subplots_adjust(left=0.07, bottom=0.13, right=0.89, top=0.94, wspace=0.15, hspace=0.25)

    errorstr = 'ptp'
    error = np.array([])
    for i in range(cP0.shape[0]):
        f = func(cP0[i][0], cP0[i][1], cP0[i][2])
        if f == 0 and abs(f - P0[i][3]) < 1e-10:
            relative_error = 0
        else:
            relative_error = abs((f - P0[i][3]) / f) * 100
        absolute_error = f - P0[i][3]
        ptp = np.ptp(node, axis=0)[3]
        ptp_error = ((f - P0[i][3]) / ptp) * 100
        errorDict = {'rel': relative_error, 'abs': absolute_error, 'ptp': ptp_error}
        error = np.append(error, errorDict[errorstr])
    error = error.reshape(int(np.sqrt(error.size)), int(np.sqrt(error.size)))
    ax1.plot_surface(X1, Y1, error, cmap='rainbow')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('ERROR')
    ax1.set_title('new method', y=-0.3)

    for i in range(5):
        list = ['IDW', 'RBF', 'MQ', 'TPS', 'Poly']
        ax1 = fig.add_subplot(3, 2, i + 2, projection='3d')
        ax1.set_title(list[i], y=-0.3)
        errorx = np.array([])
        for j in range(cP0.shape[0]):
            f = func(cP0[j][0], cP0[j][1], cP0[j][2])
            ptp = np.ptp(node, axis=0)[3]
            other = Others(cP0[j], node)
            errorx = np.append(errorx, other.compute_error(f, ptp, method=list[i], whicherror=errorstr))
            # ax1.scatter(cP0[j][0], cP0[j][1], P0x, marker='o', Alpha=1)
        errorx = errorx.reshape(int(np.sqrt(errorx.size)), int(np.sqrt(errorx.size)))
        ax1.plot_surface(X1, Y1, errorx, cmap='rainbow')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('ERROR')
        ax1.set_title(list[i], y=-0.3)

xsurface()
ysurface()
zsurface()

plt.show()
