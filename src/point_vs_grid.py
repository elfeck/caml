import numpy as np
import math
import matplotlib.pyplot as plt

points = [0.1 * math.pi,
          0.2 * math.pi,
          0.4 * math.pi,
          0.8 * math.pi]
gridpoints = [math.pi * (i / 8.0) for i in range(0, 8) if i % 2 == 1]
spacepoints = [math.pi * (i / 4.0) for i in range(0, 4)]
targets = [math.sin(x) for x in points]


step = 0.001 * math.pi
start = 0
end = math.pi + step

xval = np.arange(start, end, step)
yval = [math.sin(x) for x in xval]

savedir = "../slides/images/"

def pointBasedExample(s = True):
    f = plt.figure(1)
    sb = f.add_subplot(111)

    plt.plot(xval, yval, c="red")
    plt.plot(points, [math.sin(x) for x in points], 'bo', markersize=8)
    plt.axis([start, end, 0, 1.2])
    for x in points:
        y = math.sin(x)
        plt.plot((x, x), (0, y), c="blue")

    sb.set_xticklabels([])
    sb.set_yticklabels([])
    plt.savefig(savedir + "pointbase.png", bbox_inches="tight")
    if s:
        plt.show()


def gridBasedExample(s = True):
    f = plt.figure(1)
    sb = f.add_subplot(111)

    plt.plot(xval, yval, c="red")
    plt.plot(points, [math.sin(x) for x in points], "bo", markersize=4)
    plt.plot(gridpoints, [math.sin(x) for x in gridpoints], "go", markersize=8)
    plt.axis([start, end, 0, 1.2])
    for s in spacepoints:
        plt.plot((s, s), (0,1.2), c="green", linewidth=2)
    for g in gridpoints:
        plt.plot((g, g), (0, math.sin(g)), "--g")

    sb.set_xticklabels([])
    sb.set_yticklabels([])
    plt.savefig(savedir + "gridbase.png", bbox_inches="tight")
    if s:
        plt.show()
