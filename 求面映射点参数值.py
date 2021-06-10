import numpy as np
from 判断面映射点在多边形内 import ifPointinPoly
def getPsparameter(cPs, ns, Pe_array, sp_array):
    #Ps: [x, y, z],ns=[A,B,C],Pe_array,sp_array: [[x1, y1, z1, f1], ...,[x4, y4, z4, f4]]
    #判断Ps是否在边内部
    #计算fs
    '''
    d1 = ((x1 - xs) ** 2 + (y1 - ys) ** 2 + (z1 - zs) ** 2) ** 0.5
    d2 = ((x2 - xs) ** 2 + (y2 - ys) ** 2 + (z2 - zs) ** 2) ** 0.5
    d3 = ((x3 - xs) ** 2 + (y3 - ys) ** 2 + (z3 - zs) ** 2) ** 0.5
    d4 = ((x4 - xs) ** 2 + (y4 - ys) ** 2 + (z4 - zs) ** 2) ** 0.5

    print(Pe_array)
    temp_array1 = Pe_array[...,0:3]  #切除Pe_array的参数f值
    temp_array2 = np.subtract(temp_array1, Ps)  #计算Pe的x,y,z与Ps的插值,(x1 - xs)
    temp_array3 = np.power(temp_array2,2)  #上一步结果的平方,(x1 - xs) ** 2
    temp_array4 = np.sum(temp_array3, axis=1)  #(x1 - xs) ** 2 + (y1 - ys) ** 2 + (z1 - zs) ** 2
    d_array = np.power(temp_array4,0.5)  #((x1 - xs) ** 2 + (y1 - ys) ** 2 + (z1 - zs) ** 2) ** 0.5
    '''
    d_array = np.power(np.sum(np.power(np.subtract(Pe_array[...,0:3], cPs), 2), axis=1), 0.5)  #Pe距离数组
    f_array = Pe_array[...,3]  #Pe参数数组
    coordofsp = sp_array[...,0:3]  #映射面边界点坐标数组
    if (0 not in d_array) & ifPointinPoly(cPs, ns, coordofsp):
        reciprocalofd = np.reciprocal(d_array)  # Pe距离数组的倒数
        fs = np.sum(np.multiply(reciprocalofd, f_array)) / np.sum(reciprocalofd)
    else:
        minindex = np.argmin(d_array)
        print(minindex)
        fs = f_array[minindex]
    Ps = np.append(cPs, fs)
    return Ps
# Ps = [1,1,0]
# ns = [0,0,1]
# Pe_array = np.array([[1,0,0,1.5],[2,1,0,2.5],[1,2,0,2.5],[0,1,0,1.5]])
# sp_array = np.array([[0,0,0,1],[2,0,0,2],[2,2,0,3],[0,2,0,2]])
# Ps = getPsparameter(Ps,ns,Pe_array,sp_array)
# print(Ps)