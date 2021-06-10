import numpy as np
def surfacefunction(sp_array):
    #s_p1, s_p2, s_p3, s_p4: [x, y, z, f]
    '''
    x1 ,y1, z1 = s_p1[0], s_p1[1], s_p1[2]
    x2 ,y2, z2 = s_p2[0], s_p2[1], s_p2[2]
    x3, y3, z3 = s_p3[0], s_p3[1], s_p3[2]
    x4, y4, z4 = s_p4[0], s_p4[1], s_p4[2]

    A = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
    B = (x3 - x1) * (z2 - z1) - (x2 - x1) * (z3 - z1)
    C = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    D = -(A * x1 + B * y1 + C * z1)
    '''
    s_p1, s_p2, s_p3, s_p4 = sp_array[0], sp_array[1], sp_array[2], sp_array[3]
    point1 = np.asarray(s_p1[0:3])
    point2 = np.asarray(s_p2[0:3])
    point3 = np.asarray(s_p3[0:3])
    P1P2 = np.asmatrix(point2 - point1)
    P1P3 = np.asmatrix(point3 - point1)
    N = np.cross(P1P2, P1P3)  # 向量叉乘，求法向量
    A = N[0,0]
    B = N[0,1]
    C = N[0,2]
    D = -(A * point1[0] + B * point1[1] + C * point1[2])

    ns = [A, B, C]

    return ns, D

# ns, D = surfacefunction([1,0,7,0], [0,0,0,2], [2,0,0,3], [1,0,0,2])
# print(ns)
# print(D)