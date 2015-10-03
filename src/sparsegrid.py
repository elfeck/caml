import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

savedir = "../slides/images/"

start = 0
end = 1
step = 0.0025

f = plt.figure()

def hatfun(x, l = 0, i = 0):
    return max(1 - abs(pow(2, l) * x - i), 0)

def drawHat(l, col, layout = 111, dsb = True):
    gridPoints = [i for i in range(1, 2**l) if i % 2 == 1]
    xval = np.arange(start, end + step, step)

    if dsb:
        sb = f.add_subplot(layout)
        plt.xticks([(1/(2 * len(gridPoints))) * g for g in gridPoints])
        sb.set_xticklabels(gridPoints)
        sb.set_yticklabels([])

    ys = []
    for g in gridPoints:
        ys.append([hatfun(x, l, g) for x in xval])

    for y in ys:
        plt.plot(xval, y, color=col)
        #t = plt.text(0.01, 0.85,"l = " + str(l))
    if dsb:
        return sb

def drawHats():
    drawHat(1, "blue", 311)
    drawHat(2, "blue", 312)
    drawHat(3, "blue", 313)
    #drawHat(4, "red", 414)

    plt.savefig(savedir + "sparse_hats.png", bbox_inches="tight")
    plt.show()


def drawAllTogether():
    sb = drawHat(1, "blue",  211)
    plt.xticks([(1/8) * i for i in range(1, 9)])
    sb.set_xticklabels(range(1, 8))
    sb.set_yticklabels([])
    drawHat(2, "blue", 414, False)
    drawHat(3, "blue", 414, False)

    sb2 = f.add_subplot(212)
    xval = np.arange(start, end + step, step)
    for g in range(1,8):
        y = [hatfun(x, 3, g) for x in xval]
        plt.plot(xval, y, color="blue")

    plt.xticks([(1/8) * i for i in range(1, 9)])
    sb2.set_xticklabels(range(1, 8))
    sb2.set_yticklabels([])

    plt.savefig(savedir + "sparse_together.png", bbox_inches="tight")
    plt.show()

#drawHats()
drawAllTogether()
