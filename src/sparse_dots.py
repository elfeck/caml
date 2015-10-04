import numpy as np
import math
import matplotlib.pyplot as pltimport
from mpl_toolkits.mplot3d import Axes3D

savedir = "../slides/images/"

maxl = 3

def gridpoints(lx, ly):
    start = 0
    end = 2**maxl

    gx = [2**(maxl - lx) * i for i in range(1, 2**lx) if i % 2 == 1]
    gy = [2**(maxl - ly) * i for i in range(1, 2**ly) if i % 2 == 1]

    g = [(x, y) for x in gx for y in gy]

    return list(zip(*g))

def drawAll():
    f, axs = plt.subplots(3, 4)
    axs[0][3].axis("off")
    axs[2][3].axis("off")
    for i in range(1, maxl+1):
        for j in range(1, maxl+1):
            pts = gridpoints(i, j)
            ax = axs[j-1][i-1]
            ax.plot(pts[0], pts[1], "ob", ms=7)
            ax.axis([0, 8, 0, 8])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    for i in range(1, maxl+1):
        for j in range(1, maxl+1):
            pts = gridpoints(i, j)
            ax = axs[1][3]
            ax.set_title("Sum")
            ax.plot(pts[0], pts[1], "ob", ms=7)
            ax.axis([0, 8, 0, 8])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    plt.savefig(savedir + "sparsegrid_hirach1.png", bbox_inches="tight")
    plt.show()

def drawAll2():
    f, axs = plt.subplots(3, 4)
    axs[0][3].axis("off")
    axs[2][3].axis("off")
    for i in range(1, maxl+1):
        for j in range(1, maxl+1):
            pts = gridpoints(i, j)
            ax = axs[j-1][i-1]
            if (i + j) > 4:
                ax.plot(pts[0], pts[1], "o", ms=3, color="grey")
                ax.set_axis_bgcolor((0.9, 0.9, 0.9))
            else:
                ax.plot(pts[0], pts[1], "ob", ms=7)
            ax.axis([0, 8, 0, 8])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    for i in range(1, maxl+1):
        for j in range(1, maxl+1):
            pts = gridpoints(i, j)
            ax = axs[1][3]
            ax.set_title("Sum")
            if (i + j) <= 4:
                ax.plot(pts[0], pts[1], "ob", ms=7)
            ax.axis([0, 8, 0, 8])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    plt.savefig(savedir + "sparsegrid_hirach2.png", bbox_inches="tight")
    plt.show()

drawAll()
