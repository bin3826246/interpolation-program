import numpy as np
def getPeparameter(cPe, edge_p1, edge_p2):
    """
    @Pe: [x, y, z], edge_p1, edge_p2 : [x, y, z, f]
    """
    xe = cPe[0]
    ye = cPe[1]
    ze = cPe[2]

    x1 = edge_p1[0]
    y1 = edge_p1[1]
    z1 = edge_p1[2]
    f1 = edge_p1[3]

    x2 = edge_p2[0]
    y2 = edge_p2[1]
    z2 = edge_p2[2]
    f2 = edge_p2[3]

    flag = (x1 - xe) * (x2 - xe) + (y1 - ye) * (y2 - ye) + (z1 - ze) * (z2 - ze)  #Pe到棱边端点P1，P2的向量点积

    r1 = ((x1 - xe) ** 2 + (y1 - ye) ** 2 + (z1 - ze) ** 2) ** 0.5  #Pe到P1的距离
    r2 = ((x2 - xe) ** 2 + (y2 - ye) ** 2 + (z2 - ze) ** 2) ** 0.5  #Pe到P1的距离
    #判断棱边映射点是否在棱边内
    if flag < 0:  #点积小于0向量异向，Pe在棱边线段中
        fe = (r2 * f1 + r1 * f2) / (r1 + r2)  #求映射点参数
    else:
        if r1 < r2:
            fe = f1
        else:
            fe = f2
    Pe = np.append(cPe, fe)
    return Pe
# a = getPeparameter([1,0.5,0], [1,1,0,2], [1,-1,0,3])
# print(a)
