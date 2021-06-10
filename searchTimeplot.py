import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

grid = np.array([20, 30, 40, 50, 60, 70, 80, 90, 100])
xaxis = np.power(grid, 3)
y1 = np.array([0.03125,0.0625,0.140625,0.28125,0.5,0.84375,1.15625,1.6875,2.34375])
y2 = np.array([0.578125,1.890625,4.921875,10.296875,18.515625,39.921875,159.46875,382.578125,769.21875])
y3 = np.array([5.07E-04,8.60E-04,0.00176,0.00231,0.00376,0.005554,0.008079,0.01016,0.01244])

fig = plt.figure()
ax1 = plt.axes()
ax1.plot(xaxis, y1, 'o-')
plt.xlabel('网格（个）')
plt.ylabel('时间（s）')

fig2 = plt.figure()
ax2 = plt.axes()
ax2.plot(xaxis, y2, 'o-')
plt.xlabel('网格（个）')
plt.ylabel('时间（s）')

fig3 = plt.figure()
ax3 = plt.axes()
ax3.plot(xaxis, y3, 'o-')
plt.xlabel('网格（个）')
plt.ylabel('时间（s）')

# fig4 = plt.figure()
# ax4 = plt.axes()
# ax4.plot(xaxis, y1, 'o-')
# ax4.plot(xaxis, y2, 'o-')
# ax4.plot(xaxis, y3, 'o-')
# plt.xlabel('网格（个）')
# plt.ylabel('时间（s）')

fig5 = plt.figure()
plt.subplot(3, 1, 1)
plt.plot(xaxis, y1, 'o-')
plt.ylabel('本方法时间（s）')

plt.subplot(3, 1, 2)
plt.plot(xaxis, y2, 'o-')
plt.ylabel('穷举搜索时间（s）')

plt.subplot(3, 1, 3)
plt.plot(xaxis, y3, 'o-')
plt.xlabel('网格（个）')
plt.ylabel('二分法时间（s）')

plt.show()
