import numpy as np
import matplotlib.pyplot as plt
import time
# 强力搜索
start = time.process_time()


cP0 = np.array([0, 0.29, 0.05])
grid_num = 200
s, e = -((grid_num * 0.2) / 2 - 0.1), (grid_num * 0.2) / 2 + 0.1
xx1 = np.linspace(s, e, grid_num + 1)
yy1 = np.linspace(s, e, grid_num + 1)
zz1 = np.linspace(s, e, grid_num + 1)
X1, Y1, Z1 = np.meshgrid(xx1, yy1, zz1)

octants = np.array([[False, False, False], [True, False, False], [True, True, False], [False, True, False],
                    [False, False, True], [True, False, True], [True, True, True], [False, True, True]])

coord = np.empty((8, 3))
for i in range(8):
    # print(6)
    t = np.array([])
    for x, y, z in np.nditer((X1, Y1, Z1)):
        which_octant = lambda array: np.array([array[0] - cP0[0] > 0, array[1] - cP0[1] > 0, array[2] - cP0[2] > 0])
        if (which_octant(np.array([x, y, z])) == octants[i]).all():
            t = np.append(t, np.array([x, y, z]))
    t = t.reshape(int(t.size / 3), 3)
    d = np.sum(np.power(np.subtract(t, cP0), 2), axis=1)
    minindex = np.argmin(d)
    coord[i] = t[minindex]
# print(coord)

end = time.process_time()
print('Running time: %s Seconds' % (end - start))

# fig = plt.figure()
# ax1 = plt.axes(projection='3d')
# # ax1.scatter(X1, Y1, Z1, marker='o')
# ax1.scatter(cP0[0], cP0[1], cP0[2], marker='o')
# for i in range(coord.shape[0]):
#     ax1.scatter(coord[i][0], coord[i][1], coord[i][2], marker='^',Alpha=1, color='red')
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# plt.show()