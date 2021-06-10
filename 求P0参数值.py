import numpy as np
def getP0parameter(cP0, Ps_array):
    d_array = np.power(np.sum(np.power(np.subtract(Ps_array[..., 0:3], cP0), 2), axis=1), 0.5)  # 距离数组
    f_array = Ps_array[..., 3]  #Ps参数数组
    if 0 not in d_array:
        reciprocalofd = np.reciprocal(d_array)  # Ps距离数组的倒数
        f0 = np.sum(np.multiply(reciprocalofd, f_array)) / np.sum(reciprocalofd)
    else:
        print('x')
        minindex = np.argmin(d_array)
        f0 = f_array[minindex]
    P0 = np.append(cP0,f0)
    return P0
