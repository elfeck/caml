import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

savedir = "../slides/images/"

start = 0
end = 1
step = 0.01

def fun(x, y):
    mu = (end/2.0, end/2.0)
    var = 0.02
    return math.exp(-(pow((x - mu[0]), 2) + (pow((y - mu[1]), 2))) / (2 * var))

def hatfun(x, l = 0, i = 0):
    return max(1 - abs(pow(2, l) * x - i), 0)

def gridFun():
    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')
    ax.w_zaxis.line.set_lw(0.)
    ax.set_zticks([])

    xg = yg = np.arange(start, end + step, step)
    xm, ym = np.meshgrid(xg, yg)

    zs = np.array([fun(x,y) for x in xg for y in yg])
    zm = zs.reshape(xm.shape)

    ax.plot_surface(xm, ym, zm)
    plt.savefig(savedir + "2dgrid_1.png", bbox_inches="tight")
    plt.show()

gridPoints = np.arange(0, 8, 1)
gridXCoord = [g / len(gridPoints) for g in gridPoints]

def gridFullgrid():
    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')
    ax.w_zaxis.line.set_lw(0.)
    ax.set_zticks([])

    xg = yg = np.arange(start, end + step, step)
    xm, ym = np.meshgrid(xg, yg)

    zs = []
    alphas = []

    for i in gridPoints:
        for j in gridPoints:
            z_x = [hatfun(x, 3, i) for x in xg]
            z_y = [hatfun(y, 3, j) for y in yg]
            z = np.array([zx * zy for zx in z_x for zy in z_y])
            zs.append(np.array(z))

            xc = i / 8.0
            yc = j / 8.0
            alphas.append(fun(xc, yc))
            #zm = np.reshape(z, xm.shape)
            #ax.plot_surface(xm, ym, zm)

    sum_z = [0 for z in zs[0]]
    for i in range(len(zs)):
        sum_z = np.add(sum_z, alphas[i] * zs[i])
    zm = np.reshape(sum_z, xm.shape)
    ax.plot_surface(xm, ym, zm)
    plt.savefig(savedir + "2dgrid_2.png", bbox_inches="tight")
    plt.show()
    return

#gridFun()
gridFullgrid()
