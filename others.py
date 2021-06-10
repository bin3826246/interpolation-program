import numpy as np


class Others(object):

    def __init__(self, cP0, node):
        self.cP0 = cP0
        self.node = node


    def idw(self):
        # 反距离加权法
        d_array = np.power(np.sum(np.power(np.subtract(self.node[..., 0:3], self.cP0), 2), axis=1), 0.5)  # 距离数组
        f_array = self.node[..., 3]  # Ps参数数组
        if 0 not in d_array:
            reciprocalofd = np.reciprocal(d_array)  # **1  # Ps距离数组的倒数
            f01 = np.sum(np.multiply(reciprocalofd, f_array)) / np.sum(reciprocalofd)
        else:  # 如果有距离为0的点，则取该点值
            minindex = np.argmin(d_array)
            f01 = f_array[minindex]
        P01 = np.append(self.cP0, f01)
        # reciprocalofd = np.reciprocal(d_array)  # **4.2  # Ps距离数组的倒数
        # f01 = np.sum(np.multiply(reciprocalofd, f_array)) / np.sum(reciprocalofd)
        # P01 = np.append(self.cP0, f01)

        return P01

    def rbf1(self):
        # 最简单径向基函数插值
        Z = np.zeros((4, 4))
        D = np.ones(8)
        for i in range(3):
            for j in range(8):
                D = np.append(D, self.node[j][i])
        D = D.reshape(4, 8)
        DT = np.transpose(D)

        M = np.empty((8, 8))
        for i in range(8):
            line = []
            for j in range(8):
                d = ((self.node[i][0] - self.node[j][0]) ** 2 + (self.node[i][1] - self.node[j][1]) ** 2 + (
                        self.node[i][2] - self.node[j][2]) ** 2) ** 0.5
                line.append(d)
            M[i] = np.array(line)
        A = np.vstack((np.hstack((Z, D)), np.hstack((DT, M))))
        # print('A',A)
        P = np.asarray([0, 0, 0, 0])
        for i in range(8):
            P = np.append(P, self.node[i][3])
        P = P.reshape(12, 1)

        B = np.linalg.solve(A, P)
        X = np.array([1, self.cP0[0], self.cP0[1], self.cP0[2]])
        for j in range(8):
            d = ((self.cP0[0] - self.node[j][0]) ** 2 + (self.cP0[1] - self.node[j][1]) ** 2 + (
                    self.cP0[2] - self.node[j][2]) ** 2) ** 0.5
            X = np.append(X, d)
        # print(X)
        f02 = np.dot(X, B)
        # print('f02=', f02)
        P02 = np.append(self.cP0, f02)

        return P02

    def rbf2_mq(self):
        # 多重二次曲面
        M = np.array([])
        for i in range(8):
            line = []
            for j in range(8):
                d = 0
                for k in range(3):
                    d = (self.node[i][k] - self.node[j][k]) ** 2 + d
                d = (d + 4) ** 0.5
                line.append(d)
            M = np.append(M, line)
            # print('M', M)
        # print('M', M)
        A = M.reshape(8, 8)
        # print('A',A)
        P = np.asarray([])
        for i in range(8):
            P = np.append(P, self.node[i][3])
        P = P.reshape(8, 1)

        B = np.linalg.solve(A, P)
        # print('B',B)
        X = np.array([])
        for j in range(8):
            d = 0
            for k in range(3):
                d = (self.cP0[k] - self.node[j][k]) ** 2 + d
            d = (d + 4) ** 0.5
            X = np.append(X, d)
        # print('X', X)
        f03 = np.dot(X, B)
        # print('f03=', f03)
        P03 = np.append(self.cP0, f03)

        return P03

    def tps(self):
        A = np.empty((8, 8))
        for i in range(8):
            line = []
            for j in range(8):
                d = ((self.node[i][0] - self.node[j][0]) ** 2 + (self.node[i][1] - self.node[j][1]) ** 2 + (
                        self.node[i][2] - self.node[j][2]) ** 2) ** 0.5
                if d > 0:
                    line.append((d ** 2) * np.log10(d))
                else:
                    line.append(0)
            A[i] = np.array(line)

        P = np.array([])
        for i in range(8):
            P = np.append(P, self.node[i][3])
        P = P.reshape(8, 1)
        B = np.linalg.solve(A, P)

        X = np.array([])
        for j in range(8):
            d = ((self.cP0[0] - self.node[j][0]) ** 2 + (self.cP0[1] - self.node[j][1]) ** 2 + (
                    self.cP0[2] - self.node[j][2]) ** 2) ** 0.5
            X = np.append(X, (d ** 2) * np.log10(d))
        f04 = np.dot(X, B)
        P04 = np.append(self.cP0, f04)

    def rbf3_tps(self):
        # 薄板样条插值
        Z = np.zeros((4, 4))
        D = np.ones(8)
        for i in range(3):
            for j in range(8):
                D = np.append(D, self.node[j][i])
        D = D.reshape(4, 8)
        DT = np.transpose(D)

        M = np.empty((8, 8))
        for i in range(8):
            line = []
            for j in range(8):
                d = ((self.node[i][0] - self.node[j][0]) ** 2 + (self.node[i][1] - self.node[j][1]) ** 2 + (
                        self.node[i][2] - self.node[j][2]) ** 2) ** 0.5
                if d > 0:
                    line.append((d ** 2) * np.log10(d))
                else:
                    line.append(0)
            M[i] = np.array(line)
        # print('M', M)
        A = np.vstack((np.hstack((Z, D)), np.hstack((DT, M))))
        # print('A',A)
        P = np.asarray([0, 0, 0, 0])
        for i in range(8):
            P = np.append(P, self.node[i][3])
        P = P.reshape(12, 1)

        B = np.linalg.solve(A, P)
        # print('B',B)
        X = np.array([1, self.cP0[0], self.cP0[1], self.cP0[2]])
        for j in range(8):
            d = ((self.cP0[0] - self.node[j][0]) ** 2 + (self.cP0[1] - self.node[j][1]) ** 2 + (
                    self.cP0[2] - self.node[j][2]) ** 2) ** 0.5
            if d > 0:
                X = np.append(X, (d ** 2) * np.log10(d))
            else:
                X = np.append(X, 0)
        # print(X)
        f04 = np.dot(X, B)
        # print('f04=', f04)
        P04 = np.append(self.cP0, f04)

        return P04

    def poly(self):
        A = np.array([])
        for i in range(8):
            d = np.array([self.node[i][0], self.node[i][1], self.node[i][2], 1])
            A = np.append(A, d)
        A = A.reshape(8, 4)
        AT = np.transpose(A)
        P = np.array([])
        for i in range(8):
            P = np.append(P, self.node[i][3])
        P = P.reshape(8, 1)
        B = np.dot(np.linalg.inv(np.dot(AT, A)), np.dot(AT, P))
        X = np.array([self.cP0[0], self.cP0[1], self.cP0[2], 1])
        f05 = np.dot(X, B)
        P05 = np.append(self.cP0, f05)

        return P05

    def ips(self):
        A = np.empty((8, 24))
        for i in range(8):
            line = np.array([])
            for j in range(8):
                d = ((self.node[i][0] - self.node[j][0]) ** 2 + (self.node[i][1] - self.node[j][1]) ** 2 + (
                        self.node[i][2] - self.node[j][2]) ** 2) ** 0.5
                if d > 0:
                    line = np.append(line, np.array([1, d ** 2, (d ** 2) * np.log(d ** 2)]))
                else:
                    line = np.append(line, np.array([1, 0, 0]))
            A[i] = line
        P = np.array([])
        for i in range(8):
            P = np.append(P, self.node[i][3])
        P = P.reshape(8, 1)

        B = np.linalg.lstsq(A, P, rcond=None)[0]
        X = np.array([])
        for j in range(8):
            d = ((self.cP0[0] - self.node[j][0]) ** 2 + (self.cP0[1] - self.node[j][1]) ** 2 + (
                    self.cP0[2] - self.node[j][2]) ** 2) ** 0.5
            X = np.append(X, np.array([1, d ** 2, (d ** 2) * np.log(d ** 2)]))
        # print(X)
        f06 = np.dot(X, B)
        # print('f04=', f04)
        P06 = np.append(self.cP0, f06)

    def compute_error(self, f, ptp, method, whicherror):
        dic = {'IDW': self.idw, 'RBF': self.rbf1, 'MQ': self.rbf2_mq, 'TPS': self.rbf3_tps, 'Poly':self.poly}
        P0 = dic[method]()
        if f == 0 and abs(f - P0[3]) < 1:
            relative_error,absolute_error, ptp_error = 0,0,0
        else:
            relative_error = abs((f - P0[3]) / f)*100
            absolute_error = f - P0[3]
            #ptp = np.ptp(self.node, axis=0)[3]
            ptp_error = ((f - P0[3]) / ptp)*100
        errorDict = {'rel': relative_error, 'abs': absolute_error, 'ptp': ptp_error}
        error = errorDict[whicherror]
        # print(
        #     '{}P05 = {}\n相对误差：{}\n绝对误差：{:.4f}\nptp = {:.4f}'.format(method, P0, relative_error,
        #                                                                absolute_error,
        #                                                                ptp_error))

        return error

    def compute(self, method):
        dic = {'IDW': self.idw, 'RBF': self.rbf1, 'MQ': self.rbf2_mq, 'TPS': self.rbf3_tps, 'Poly': self.poly}
        P0 = dic[method]()
        return P0
