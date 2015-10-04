import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

savedir = "../slides/images/"

start = 0
end = 1
step = 0.0025
maxl = 3

xval = yval = np.arange(start, end + step, step)
xmesh, ymesh = np.meshgrid(xval, yval)

def gridpoints(lx, ly):
    gx = [i for i in range(1, 2**lx) if i % 2 == 1]
    gy = [i for i in range(1, 2**ly) if i % 2 == 1]

    g = [(x, y) for x in gx for y in gy]

    return list(zip(*g))

def hatfun(x, l = 0, i = 0):
    return max(1 - abs(pow(2, l) * x - i), 0)


def fun(x, y):
    mu = (end/2.0, end/2.0)
    var = 0.02
    return math.exp(-(pow((x - mu[0]), 2) + (pow((y - mu[1]), 2))) / (2 * var))


def getSubspace(lx, ly):
    pts = gridpoints(lx, ly)
    hatx = []
    haty = []
    for g in pts[0]:
        hatx.append([hatfun(x, lx, g) for x in xval])
    for g in pts[1]:
        haty.append([hatfun(y, ly, g) for y in yval])

    zs = []
    for i in range(len(pts[0])):
        for j in range(len(pts[1])):
            xpos = (1/8.0) * 2**(maxl - lx) * pts[0][i]
            ypos = (1/8.0) * 2**(maxl - ly) * pts[1][j]
            alpha = fun(xpos, ypos)
            print(str(pts[0][i]) + "=" + str(xpos) + " | " +
                  str(pts[1][j]) + "=" + str(ypos) + " | alpha=" + str(alpha))
            zs.append((pts[0][i], pts[1][j],
                       [x * y for x in hatx[i] for y in haty[j]], alpha))
    return zs

def sparseDiscr():
    f = plt.figure()
    zmeshes = []
    k = 1
    totalsum = []
    totalsumSG = []
    for i in range(1, maxl+1):
        for j in range(1, maxl+1):
            s = getSubspace(i, j)
            zsum = np.zeros(len(s[0][2]))
            for hat in s:
                gx, gy, zs, alpha = hat
                zsum += np.multiply(alpha * 1/(i*j), zs)
                if len(totalsum) == 0:
                    totalsum = np.zeros(len(zsum))
                    totalsumSG = np.zeros(len(zsum))
            zmesh = np.reshape(zsum, xmesh.shape)
            totalsum += zsum
            ax = f.add_subplot(3, 4, k, projection="3d")
            k+=1
            if k % 4 == 0:
                k+=1
            if i + j <= 4:
                ax.plot_surface(xmesh, ymesh, zmesh, cmap=cm.coolwarm,
                                linewidth=0)
                totalsumSG += zsum
            else:
                ax.plot_surface(xmesh, ymesh, zmesh, cmap=cm.Greys,
                                linewidth=0)
            #ax.axis("off")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zlim(0, 1)
    ax = f.add_subplot(3, 4, 4, projection="3d")
    ax.plot_surface(xmesh, ymesh, np.reshape(
        [fun(x, y) for x in xval for y in yval], xmesh.shape), linewidth=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax = f.add_subplot(3, 4, 8, projection="3d")
    ax.plot_surface(xmesh, ymesh, np.reshape(totalsum, xmesh.shape),
                    linewidth=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax = f.add_subplot(3, 4, 12, projection="3d")
    ax.plot_surface(xmesh, ymesh, np.reshape(totalsumSG, xmesh.shape),
                    linewidth=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.show()


sparseDiscr()
