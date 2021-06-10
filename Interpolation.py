import numpy as np


class Interpolation(object):
    Surface = np.array([[0, 1, 2, 3], [0, 1, 5, 4], [1, 2, 6, 5],
                        [2, 3, 7, 6], [0, 3, 7, 4], [4, 5, 6, 7]])

    def __init__(self, cP0, node):
        self.cP0 = cP0  # [x,y,z]
        self.node = node  # [x,y,z,f] (8,4)数组

    def getEdgeFootPoint(self, cP0, edge_p1, edge_p2):
        # P0, edge_p1, edge_p2 : [x, y, z] narray_like
        x0, y0, z0 = cP0[0], cP0[1], cP0[2]
        x1, y1, z1 = edge_p1[0], edge_p1[1], edge_p1[2]
        x2, y2, z2 = edge_p2[0], edge_p2[1], edge_p2[2]
        # k = |P1Pe|/|P1P2| = |P1P0|*|P1P2|*cosθ/|P1P2|^2
        k = ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1) + (z0 - z1) * (z2 - z1)) / \
            ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) * 1.0
        xpe = k * (x2 - x1) + x1
        ype = k * (y2 - y1) + y1
        zpe = k * (z2 - z1) + z1
        cPe = np.array([xpe, ype, zpe])

        return cPe

    def getPeParameter(self, cPe, edge_p1, edge_p2):
        # Pe: [x, y, z], edge_p1, edge_p2 : [x, y, z, f] narray_like
        xe, ye, ze = cPe[0], cPe[1], cPe[2]
        x1, y1, z1, f1 = edge_p1[0], edge_p1[1], edge_p1[2], edge_p1[3]
        x2, y2, z2, f2 = edge_p2[0], edge_p2[1], edge_p2[2], edge_p2[3]

        flag = (x1 - xe) * (x2 - xe) + (y1 - ye) * (y2 - ye) + (z1 - ze) * (z2 - ze)  # Pe到棱边端点P1，P2的向量点积

        r1 = ((x1 - xe) ** 2 + (y1 - ye) ** 2 + (z1 - ze) ** 2) ** 0.5  # Pe到P1的距离
        r2 = ((x2 - xe) ** 2 + (y2 - ye) ** 2 + (z2 - ze) ** 2) ** 0.5  # Pe到P1的距离
        # 判断棱边映射点是否在棱边内
        if flag < 0:  # 点积小于0向量异向，Pe在棱边线段中
            fe = (r2 * f1 + r1 * f2) / (r1 + r2)  # 求映射点参数
        else:
            if r1 < r2:
                fe = f1
            else:
                fe = f2
        Pe = np.append(cPe, fe)

        return Pe

    def surfaceFunction(self, sp_array):
        # 求平面方程  s_p1, s_p2, s_p3, s_p4: [x, y, z, f]
        s_p1, s_p2, s_p3, s_p4 = sp_array[0], sp_array[1], sp_array[2], sp_array[3]
        point1 = np.asarray(s_p1[0:3])
        point2 = np.asarray(s_p2[0:3])
        point3 = np.asarray(s_p3[0:3])
        P1P2 = np.asmatrix(point2 - point1)
        P1P3 = np.asmatrix(point3 - point1)
        N = np.cross(P1P2, P1P3)  # 向量叉乘，求法向量
        A = N[0, 0]
        B = N[0, 1]
        C = N[0, 2]
        D = -(A * point1[0] + B * point1[1] + C * point1[2])
        ns = [A, B, C]

        return ns, D

    def getSurfaceFootPoint(self, cP0, sp_array):
        # 建立平面方程Ax+By+Cz+D=0
        s_p1, s_p2, s_p3, s_p4 = sp_array[0], sp_array[1], sp_array[2], sp_array[3]
        x1, y1, z1 = s_p1[0], s_p1[1], s_p1[2]
        x2, y2, z2 = s_p2[0], s_p2[1], s_p2[2]
        x3, y3, z3 = s_p3[0], s_p3[1], s_p3[2]
        x4, y4, z4 = s_p4[0], s_p4[1], s_p4[2]
        A = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
        B = (x3 - x1) * (z2 - z1) - (x2 - x1) * (z3 - z1)
        C = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        D = -(A * x1 + B * y1 + C * z1)
        # 求映射点坐标
        x0, y0, z0 = cP0[0], cP0[1], cP0[2]
        # 向量PsP0和P1P2的点乘为0，向量PsP0和P1P3的点乘为0
        a = np.array([[x2 - x1, y2 - y1, z2 - z1], [x3 - x1, y3 - y1, z3 - z1], [A, B, C]])
        b = np.array([[x0 * (x2 - x1) + y0 * (y2 - y1) + z0 * (z2 - z1)],
                      [x0 * (x3 - x1) + y0 * (y3 - y1) + z0 * (z3 - z1)],
                      [-D]])
        cPs = np.linalg.solve(a, b).reshape(3, )

        return cPs

    def ifRayIntersectsSegment(self, cPs, ns, edge_p1, edge_p2):
        # 射线法判断
        xs, ys, zs = cPs[0], cPs[1], cPs[2]
        if np.linalg.norm(np.cross(ns, [1, 0, 0])) == 0.0:  # 映射面于平面yoz平行时，射平面y=ys
            # 向z大于0的方向做射平面
            A = np.array([[edge_p2[1] - edge_p1[1], edge_p1[0] - edge_p2[0], 0],
                          [0, edge_p2[2] - edge_p1[2], edge_p1[1] - edge_p2[1]],
                          [0, 1, 0]])
            B = np.array([[edge_p1[0] * edge_p2[1] - edge_p2[0] * edge_p1[1]],
                          [edge_p1[1] * edge_p2[2] - edge_p2[1] * edge_p1[2]],
                          [ys]])
            aa = 1
        else:  # 映射面其他状态时，射平面x=xs
            # 向z大于0或y大于0的方向做射平面
            A = np.array([[edge_p2[1] - edge_p1[1], edge_p1[0] - edge_p2[0], 0],
                          [edge_p2[2] - edge_p1[2], 0, edge_p1[0] - edge_p2[0]],
                          [1, 0, 0]])
            B = np.array([[edge_p1[0] * edge_p2[1] - edge_p2[0] * edge_p1[1]],
                          [edge_p1[0] * edge_p2[2] - edge_p2[0] * edge_p1[2]],
                          [xs]])
            aa = 0
        if edge_p1[aa] == edge_p2[aa]:  # 排除与射线平行、重合，线段首尾端点重合的情况
            return False
        if edge_p1[aa] > cPs[aa] and edge_p2[aa] > cPs[aa]:  # 线段在射线上边
            return False
        if edge_p1[aa] < cPs[aa] and edge_p2[aa] < cPs[aa]:  # 线段在射线下边
            return False
        if edge_p1[aa] == cPs[aa] and edge_p2[aa] > cPs[aa]:  # 交点为下端点，对应point1
            return False
        if edge_p2[aa] == cPs[aa] and edge_p1[aa] > cPs[aa]:  # 交点为下端点，对应point2
            return False
        if np.vdot(ns, [0, 0, 1]) == 0.0:  # 映射面垂直于平面xoy时
            if edge_p1[2] < cPs[2] and edge_p2[2] < cPs[2]:  # 若棱边点z坐标小于zs则排除
                return False
        else:  # 其他情况
            if edge_p1[1] < cPs[1] and edge_p2[1] < cPs[1]:  # 若棱边点y坐标小于ys则排除
                return False
        # 求交点
        pointofintersection = np.linalg.solve(A, B).reshape(3, )
        if np.vdot(ns, [0, 0, 1]) == 0.0 and pointofintersection[2] <= zs:  # 垂直于平面xoy的平面判断zs
            return False
        if np.vdot(ns, [0, 0, 1]) != 0.0 and pointofintersection[1] <= ys:  # 只要不垂直于xoy都判断zs
            return False
        return True

    def ifPointInPoly(self, cPs, ns, coordofsp):
        # 输入：面映射点，映射面法向量，四边形各点坐标
        # Ps=[x,y,z],ns=[A,B,C],coordofsp=[[x1,y1,z1],[x2,y2,z2],[x3,y3,z3],[x4,y4,z4]] narray_like
        poly = np.append(coordofsp, coordofsp[0:1],
                         axis=0)  # poly=[[x1,y1,z1],[x2,y2,z2],[x3,y3,z3],[x4,y4,z4],[x1,y1,z1]]
        sinsc = 0  # 交点个数
        for i in range(len(poly) - 1):  # range(4)
            # 循环每条边的曲线
            edge_p1 = poly[i]
            edge_p2 = poly[i + 1]
            if self.ifRayIntersectsSegment(cPs, ns, edge_p1, edge_p2):
                sinsc += 1  # 有交点就加1
        if sinsc % 2 == 1:
            return True
        else:
            return False

    def getPsParameter(self, cPs, ns, Pe_array, sp_array):
        # Ps: [x, y, z],ns=[A,B,C],Pe_array,sp_array: [[x1, y1, z1, f1], ...,[x4, y4, z4, f4]]
        # 判断Ps是否在边内部,计算fs
        # temp_array1 = Pe_array[...,0:3]  #切除Pe_array的参数f值
        # temp_array2 = np.subtract(temp_array1, Ps)  #计算Pe的x,y,z与Ps的插值,(x1 - xs)
        # temp_array3 = np.power(temp_array2,2)  #上一步结果的平方,(x1 - xs) ** 2
        # temp_array4 = np.sum(temp_array3, axis=1)  #(x1 - xs) ** 2 + (y1 - ys) ** 2 + (z1 - zs) ** 2
        # d_array = np.power(temp_array4,0.5)  #((x1 - xs) ** 2 + (y1 - ys) ** 2 + (z1 - zs) ** 2) ** 0.5
        d_array = np.power(np.sum(np.power(np.subtract(Pe_array[..., 0:3], cPs), 2), axis=1), 0.5)  # Pe距离数组
        f_array = Pe_array[..., 3]  # Pe参数数组
        coordofsp = sp_array[..., 0:3]  # 映射面边界点坐标数组
        if (0 not in d_array) & self.ifPointInPoly(cPs, ns, coordofsp):
            reciprocalofd = np.reciprocal(d_array)  # **15  # Pe距离数组的倒数
            fs = np.sum(np.multiply(reciprocalofd, f_array)) / np.sum(reciprocalofd)
        else:  # 如果有距离为0的点，则取该点值
            minindex = np.argmin(d_array)
            fs = f_array[minindex]
        Ps = np.append(cPs, fs)

        return Ps

    def getP0Parameter(self, cP0, Ps_array):
        d_array = np.power(np.sum(np.power(np.subtract(Ps_array[..., 0:3], cP0), 2), axis=1), 0.5)  # 距离数组
        f_array = Ps_array[..., 3]  # Ps参数数组
        if 0 not in d_array:
            reciprocalofd = np.reciprocal(d_array)  # **1  # Ps距离数组的倒数
            f0 = np.sum(np.multiply(reciprocalofd, f_array)) / np.sum(reciprocalofd)
        else:  # 如果有距离为0的点，则取该点值
            minindex = np.argmin(d_array)
            f0 = f_array[minindex]
        P0 = np.append(cP0, f0)
        return P0

    def compute(self):
        P0 = np.array([])
        for cP0i in range(self.cP0.shape[0]):
            Ps_array = []
            for i in range(6):
                # 循环面，求面映射点
                sp_array = self.node[self.Surface[i]]
                spi = np.append(sp_array, sp_array[0:1], axis=0)
                Pe_array = []
                for j in range(4):
                    # 循环边，求边映射点
                    edge_p1 = spi[j]
                    edge_p2 = spi[j + 1]
                    Pe = self.getPeParameter(self.getEdgeFootPoint(self.cP0[cP0i], edge_p1, edge_p2), edge_p1, edge_p2)
                    Pe_array = np.append(Pe_array, Pe, axis=0)
                Pe_array = Pe_array.reshape(4, 4)
                ns, D = self.surfaceFunction(sp_array)
                cPs = self.getSurfaceFootPoint(self.cP0[cP0i], sp_array)
                Ps = self.getPsParameter(cPs, ns, Pe_array, sp_array)
                Ps_array = np.append(Ps_array, Ps, axis=0)
            Ps_array = Ps_array.reshape(6, 4)
            P0 = np.append(P0, self.getP0Parameter(self.cP0[cP0i], Ps_array))
        P0 = P0.reshape(int(P0.size / 4), 4)

        return P0
