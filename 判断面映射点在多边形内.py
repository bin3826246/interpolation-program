import numpy as np
def ifRayIntersectsSegment(cPs, ns, edge_p1, edge_p2):
    #射线法判断
    xs, ys, zs = cPs[0], cPs[1], cPs[2]
    if np.linalg.norm(np.cross(ns, [1, 0, 0])) == 0.0:  #映射面于平面yoz平行时，射平面y=ys
        # 向z大于0的方向做射平面
        A = np.array([[edge_p2[1] - edge_p1[1], edge_p1[0] - edge_p2[0], 0],
                      [0, edge_p2[2] - edge_p1[2], edge_p1[1] - edge_p2[1]],
                      [0, 1, 0]])
        B = np.array([[edge_p1[0] * edge_p2[1] - edge_p2[0] * edge_p1[1]],
                      [edge_p1[1] * edge_p2[2] - edge_p2[1] * edge_p1[2]],
                      [ys]])
        #print(999)
        aa = 1
    else:  #映射面其他状态时，射平面x=xs
        # 向z大于0或y大于0的方向做射平面
        A = np.array([[edge_p2[1] - edge_p1[1], edge_p1[0] - edge_p2[0], 0],
                      [edge_p2[2] - edge_p1[2], 0, edge_p1[0] - edge_p2[0]],
                      [1, 0, 0]])
        B = np.array([[edge_p1[0] * edge_p2[1] - edge_p2[0] * edge_p1[1]],
                      [edge_p1[0] * edge_p2[2] - edge_p2[0] * edge_p1[2]],
                      [xs]])
        #print(888)
        aa = 0
    if edge_p1[aa] == edge_p2[aa]: #排除与射线平行、重合，线段首尾端点重合的情况
        return False
    if edge_p1[aa] > cPs[aa] and edge_p2[aa] > cPs[aa]: #线段在射线上边
        return False
    if edge_p1[aa] < cPs[aa] and edge_p2[aa] < cPs[aa]: #线段在射线下边
        return False
    if edge_p1[aa] == cPs[aa] and edge_p2[aa] > cPs[aa]: #交点为下端点，对应point1
        return False
    if edge_p2[aa] == cPs[aa] and edge_p1[aa] > cPs[aa]: #交点为下端点，对应point2
        return False
    if np.vdot(ns, [0, 0, 1]) == 0.0:  #映射面垂直于平面xoy时
        if edge_p1[2] < cPs[2] and edge_p2[2] < cPs[2]:  #若棱边点z坐标小于zs则排除
            return False
    else:  #其他情况
        if edge_p1[1] < cPs[1] and edge_p2[1] < cPs[1]:  # 若棱边点y坐标小于ys则排除
            return False
    # 求交点
    pointofintersection = np.linalg.solve(A, B).reshape(3,)
    if np.linalg.norm(np.cross(ns, [0, 0, 1])) == 0.0 and pointofintersection[1] <= ys:  #平行于平面xoy的平面判断ys
        return False
    if np.linalg.norm(np.cross(ns, [0, 0, 1])) != 0.0 and pointofintersection[2] <= zs:  #只要不平行于xoy都判断zs
        return False
    return True
def ifPointinPoly(cPs, ns, coordofsp):
    #输入：面映射点，映射面法向量，四边形各点坐标
    #Ps=[x,y,z],ns=[A,B,C],coordofsp=[[x1,y1,z1],[x2,y2,z2],[x3,y3,z3],[x4,y4,z4]]
    poly = np.append(coordofsp, coordofsp[0:1], axis=0)  #poly=[[x1,y1,z1],[x2,y2,z2],[x3,y3,z3],[x4,y4,z4],[x1,y1,z1]]
    sinsc=0 #交点个数
    for i in range(len(poly)-1):   #range(4)
        #循环每条边的曲线
        edge_p1 = poly[i]
        edge_p2 = poly[i+1]
        if ifRayIntersectsSegment(cPs, ns, edge_p1, edge_p2):
            sinsc += 1 #有交点就加1
    if sinsc % 2 == 1:
        return True
    else:
        return False
# Ps = [1,1,0]
# ns = [0,0,1]
# coordofsp = [[0,0,0],[2,0,0],[2,2,0],[0,2,0]]
# if ifPointinPoly(Ps, ns, coordofsp):
#     print(1)
# else:
#     print(0)
#
# Ps = [0,1,1]
# ns = [1,0,0]
# coordofsp = [[0,0,0],[0,2,0],[0,2,2],[0,0,2]]
# if ifPointinPoly(Ps, ns, coordofsp):
#     print(1)
# else:
#     print(0)
#
# Ps = [1,0,1]
# ns = [0,1,0]
# coordofsp = [[0,0,0],[0,0,2],[2,0,2],[2,0,0]]
# if ifPointinPoly(Ps, ns, coordofsp):
#     print(1)
# else:
#     print(0)