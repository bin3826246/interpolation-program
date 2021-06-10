import numpy as np
import matplotlib.pyplot as plt
import time

fig = plt.figure()
ax1 = plt.axes(projection='3d')
start = time.process_time()
# 生成三维离散点
grid_side_length = 0.2
grids_num = 15
start_coordinate, end_coordinate = -((grids_num * grid_side_length) / 2 - 0.1), (grids_num * grid_side_length) / 2 + 0.1
xx1 = np.linspace(start_coordinate, end_coordinate, grids_num + 1)
yy1 = np.linspace(start_coordinate, end_coordinate, grids_num + 1)
zz1 = np.linspace(start_coordinate, end_coordinate, grids_num + 1)
X1, Y1, Z1 = np.meshgrid(xx1, yy1, zz1)
# 构建桶
bucket_length = grid_side_length
cP0 = np.array([0, 0.29, 0.05])
bucket = np.array([])
for x, y, z in np.nditer((X1, Y1, Z1)):
    # if x**2+y**2 <= 1:
    #     ax1.scatter(x,y,z, marker='o',alpha=0.6)
    if abs(x - cP0[0]) <= bucket_length and abs(y - cP0[1]) <= bucket_length and abs(z - cP0[2]) <= bucket_length:
        bucket = np.append(bucket, np.array([x, y, z]))
bucket = bucket.reshape(int(bucket.size / 3), 3)

octants = np.array([[False, False, False], [True, False, False], [True, True, False], [False, True, False],
                    [False, False, True], [True, False, True], [True, True, True], [False, True, True]])
coord = np.empty((8, 3))
for i in range(8):
    t = np.array([])
    for j in range(bucket.shape[0]):
        # 确定卦限
        which_octant = lambda array: np.array([array[0] - cP0[0] > 0, array[1] - cP0[1] > 0, array[2] - cP0[2] > 0])
        if (which_octant(bucket[j]) == octants[i]).all():
            t = np.append(t, bucket[j])
    t = t.reshape(int(t.size / 3), 3)
    # 构建插值网格单元
    d = np.sum(np.power(np.subtract(t, cP0), 2), axis=1)
    minindex = np.argmin(d)
    coord[i] = t[minindex]
# print(coord)

end = time.process_time()
print('Running time: %s Seconds' % (end - start))

# fig = plt.figure()
# ax1 = plt.axes(projection='3d')
ax1.scatter(X1, Y1, Z1, marker='o')

ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
ax1.set_zlim(-2, 2)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
plt.show()
