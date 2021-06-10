import numpy as np

def getsurfaceFootpoint(cP0, sp_array):
    #建立平面方程Ax+By+Cz+D=0
    s_p1, s_p2, s_p3, s_p4 = sp_array[0], sp_array[1], sp_array[2], sp_array[3]
    x1, y1, z1 = s_p1[0], s_p1[1], s_p1[2]
    x2, y2, z2 = s_p2[0], s_p2[1], s_p2[2]
    x3, y3, z3 = s_p3[0], s_p3[1], s_p3[2]
    x4, y4, z4 = s_p4[0], s_p4[1], s_p4[2]
    A = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
    B = (x3 - x1) * (z2 - z1) - (x2 - x1) * (z3 - z1)
    C = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    D = -(A * x1 + B * y1 + C * z1)
    #求映射点坐标
    x0, y0, z0 = cP0[0], cP0[1], cP0[2]
    #向量PsP0和P1P2的点乘为0，向量PsP0和P1P3的点乘为0
    a = np.array([[x2 - x1, y2 - y1, z2 - z1], [x3 - x1, y3 - y1, z3 - z1], [A, B, C]])
    b = np.array([[x0 * (x2 - x1) + y0 * (y2 - y1) + z0 * (z2 - z1)],
                 [x0 * (x3 - x1) + y0 * (y3 - y1) + z0 * (z3 - z1)],
                 [-D]])
    cPs = np.linalg.solve(a, b).reshape(3,)
    return cPs
# a = getsurfaceFootpoint([1,1,1], [[1,1,0,0], [0,0,0,2], [2,3,0,3], [1,0,0,2]])
# print(a)

