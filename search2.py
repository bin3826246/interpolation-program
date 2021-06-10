import numpy as np
import matplotlib.pyplot as plt
import time
# 二分法
# start = time.process_time()
start = time.perf_counter()
cP0 = np.array([0, 0.29, 0.05])
grid_num = 70
s, e = -((grid_num * 0.2) / 2 - 0.1), (grid_num * 0.2) / 2 + 0.1
xx1 = np.linspace(s, e, grid_num + 1)
yy1 = np.linspace(s, e, grid_num + 1)
zz1 = np.linspace(s, e, grid_num + 1)
X1, Y1, Z1 = np.meshgrid(xx1, yy1, zz1)

# print(xx1.shape[0])
# time.sleep(1)
xyz = np.empty((3,2))
i = 0
for co in [xx1, yy1, zz1]:
    index1 = 0
    index2 = co.shape[0] - 1
    while round(abs(co[index1] - co[index2]), 1) > 0.2:
        # print(i)
        mid = (co[index1] + co[index2]) / 2
        midindex = (index1 + index2) / 2
        if cP0[i] > mid:
            index1 = int(midindex)
        elif cP0[i] < mid:
            index2 = int(round(midindex, 0))
        else:
            index1 = int(midindex)
            index2 = index1 + 1
            break
    xyz[i] = np.array([co[index1], co[index2]])
    i = i + 1

# print(xyz)

coord = np.array([[xyz[0][0], xyz[1][0], xyz[2][0]],
                  [xyz[0][1], xyz[1][0], xyz[2][0]],
                  [xyz[0][1], xyz[1][1], xyz[2][0]],
                  [xyz[0][0], xyz[1][1], xyz[2][0]],
                  [xyz[0][0], xyz[1][0], xyz[2][1]],
                  [xyz[0][1], xyz[1][0], xyz[2][1]],
                  [xyz[0][1], xyz[1][1], xyz[2][1]],
                  [xyz[0][0], xyz[1][1], xyz[2][1]],
                  ])
# print(coord)

# end = time.process_time()
end = time.perf_counter()
print('Running time: {:.6f} Seconds'.format(end - start))

# fig = plt.figure()
# ax1 = plt.axes(projection='3d')
# # ax1.scatter(X1, Y1, Z1, marker='o')
# ax1.scatter(cP0[0], cP0[1], cP0[2], marker='o')
# for i in range(coord.shape[0]):
#     ax1.scatter(coord[i][0], coord[i][1], coord[i][2], marker='^',Alpha=1, color='red')
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# plt.show()