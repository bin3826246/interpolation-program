import numpy as np
def getedgeFootPoint(cP0, edge_p1, edge_p2):
    """
    P0, edge_p1, edge_p2 : [x, y, z]
    """
    x0 = cP0[0]
    y0 = cP0[1]
    z0 = cP0[2]

    x1 = edge_p1[0]
    y1 = edge_p1[1]
    z1 = edge_p1[2]

    x2 = edge_p2[0]
    y2 = edge_p2[1]
    z2 = edge_p2[2]
    #k = P1Pe/P1P2
    k = ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1) + (z0 - z1) * (z2 - z1)) / \
        ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)*1.0

    xpe = k * (x2 - x1) + x1
    ype = k * (y2 - y1) + y1
    zpe = k * (z2 - z1) + z1

    cPe= np.asarray([xpe, ype, zpe])
    return cPe

# a = getedgeFootPoint([0,0,0], [1,1,0], [2,-1,0])
# print(a)
