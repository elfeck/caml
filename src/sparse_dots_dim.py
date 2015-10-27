import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

savedir = "../slides/images/"

maxl = 4

def gridpoints(lx, ly):
    start = 0
    end = 2**maxl

    gx = [2**(maxl - lx) * i for i in range(1, 2**lx) if i % 2 == 1]
    gy = [2**(maxl - ly) * i for i in range(1, 2**ly) if i % 2 == 1]

    g = [(x, y) for x in gx for y in gy]

    return list(zip(*g))

def drawAll2():
    f, ax = plt.subplots(1, 1)
    for i in range(1, maxl+1):
        for j in range(1, maxl+1):
            pts = gridpoints(i, j)
            if (i + j) < 2 + maxl:
                ax.plot(pts[0], pts[1], "ob", ms=7)
            ax.axis([0, 2**maxl, 0, 2**maxl])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    plt.savefig(savedir + "sparsegrid_d4.png", bbox_inches="tight")
    plt.show()

drawAll2()
