import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#from mpl_toolkits.mplot3d import proj3d

#def orthogonal_proj(zfront, zback):
#    a = (zfront+zback)/(zfront-zback)
#    b = -2*(zfront*zback)/(zfront-zback)
#    return np.array([[1,0,0,0],
#                        [0,1,0,0],
#                        [0,0,a,b],
#                        [0,0,0,zback]])
#proj3d.persp_transformation = orthogonal_proj

savedir = "../slides/images/"

start = 0
end = 1
step = 0.0025
maxl = 3


xval = yval = np.arange(start, end + step, step)
xmesh, ymesh = np.meshgrid(xval, yval)

def gridpoints(lx, ly):
    start = 0
    end = 2**maxl

    gx = [i for i in range(1, 2**lx) if i % 2 == 1]
    gy = [i for i in range(1, 2**ly) if i % 2 == 1]

    g = [(x, y) for x in gx for y in gy]

    return list(zip(*g))

def hatfun(x, l = 0, i = 0):
    return max(1 - abs(pow(2, l) * x - i), 0)

def get2dHats(lx, ly):
    pts = gridpoints(lx, ly)
    hatx = []
    haty = []
    for g in pts[0]:
        hatx.append([hatfun(x, lx, g) for x in xval])
    for g in pts[1]:
        haty.append([hatfun(y, ly, g) for y in yval])

    zs = [[x * y for x in cx for y in cy] for cx in hatx for cy in haty]
    zval = np.zeros(len(zs[0]))
    for z in zs:
        zval = zval + z

    zmesh = np.reshape(np.array(zval), xmesh.shape)
    return zmesh


def drawAll():
    f = plt.figure()
    k = 1
    maxl = 3
    print("start")
    for i in range(1, maxl+1):
        for j in range(1, maxl+1):
            zmesh = get2dHats(i, j)
            ax = f.add_subplot(maxl, maxl, k, projection="3d")
            k+=1
            ax.plot_surface(xmesh, ymesh, zmesh, cmap=cm.coolwarm, linewidth=0)
            #ax.axis("off")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

    plt.savefig(savedir + "sparsegrid_2dhats.png", bbox_inches="tight")
    plt.show()

drawAll()
