import numpy as np
import matplotlib.pyplot as plt
import time

fig = plt.figure()
ax1 = plt.axes(projection='3d')
start = time.process_time()
# 生成三维离散点
grid_side_length = 0.2
grids_num = 20
start_coordinate, end_coordinate = -((grids_num * grid_side_length) / 2 - 0.1), (grids_num * grid_side_length) / 2 + 0.1
xx1 = np.linspace(start_coordinate, end_coordinate, grids_num + 1)

X = np.array([])
Y = np.array([])
Z = np.array([])
for i in range(grids_num):
    yy1 = np.linspace(start_coordinate+grid_side_length/4*i, end_coordinate+grid_side_length/4*i, grids_num + 1)
    zz1 = np.linspace(start_coordinate+i, start_coordinate+i, grids_num + 1)
    X1, Y1, Z1 = np.meshgrid(xx1, yy1, zz1)
    X = np.append(X, X1)
    X = X.reshape(int(np.cbrt(X.size)), int(np.cbrt(X.size)),int(np.cbrt(X.size)))
    Y = np.append(Y, Y1)
    Y = Y.reshape(int(np.cbrt(Y.size)), int(np.cbrt(Y.size)), int(np.cbrt(Y.size)))
    Z = np.append(Z, Z1)
    Z = Z.reshape(int(np.sqrt(Z.size)), int(np.sqrt(Z.size)), int(np.cbrt(Y.size)))

# 构建桶
bucket_length = grid_side_length
cP0 = np.array([0, 0.29, 0.05])
bucket = np.array([])
for x, y, z in np.nditer((X, Y, Z)):
    if x**2+y**2 <= 1:
        ax1.scatter(x,y,z, marker='o',alpha=0.6)

end = time.process_time()
print('Running time: %s Seconds' % (end - start))

# fig = plt.figure()
# ax1 = plt.axes(projection='3d')
# ax1.scatter(X1, Y1, Z1, marker='o')

ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
ax1.set_zlim(-2, 2)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
plt.show()
