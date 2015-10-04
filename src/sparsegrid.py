import numpy as np
import math
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

start = 0
end = 1
step = 0.01
ml = 3

def fun(x, y):
    mu = (end/2.0, end/2.0)
    var = 0.02
    return math.exp(-(pow((x - mu[0]), 2) +
                      (pow((y - mu[1]), 2))) / (2 * var))

class SparseGrid:

    def __init__(self):
        subs = [Subspace(1, 1, None)]         # 0: 11
        subs.append(Subspace(1,2,subs[0]))    # 1: 12
        subs.append(Subspace(1,3,subs[1]))    # 2: 13

        subs.append(Subspace(2,1,subs[0]))    # 3: 21
        subs.append(Subspace(2,2,subs[0]))    # 4: 22
        subs.append(Subspace(2,3,subs[2]))    # 5: 23

        subs.append(Subspace(3,1,subs[3]))
        subs.append(Subspace(3,2,subs[3]))
        subs.append(Subspace(3,3,subs[4]))

        xval = yval = np.arange(start, end + step, step)
        xmesh, ymesh = np.meshgrid(xval, yval)
        zvals = [[x * y
                 for x in s.get_sum_x(xval)
                 for y in s.get_sum_y(yval)]
                 for s in subs]

        f = plt.figure()

        k = 1
        for zs in zvals:
            ax = f.add_subplot(3, 4, k, projection="3d")
            k+=1
            if k % 4 == 0:
                k+=1
            zmesh = np.reshape(zs, xmesh.shape)
            ax.plot_surface(xmesh, ymesh, zmesh, cstride=2, rstride=2)

        zval = zvals[0]
        for zs in zvals:
            zval = np.add(zval, zs)
            zmesh = np.reshape(zval, xmesh.shape)
            ax = f.add_subplot(3, 4, 8, projection="3d")
            ax.plot_surface(xmesh, ymesh, zmesh, cstride=5, rstride=5)
        plt.show()

class Subspace:

    def __init__(self, lx, ly, parent):
        self.lx = lx
        self.ly = ly
        self.gxs = [i for i in range(1, 2**lx) if i % 2 == 1]
        self.gys = [i for i in range(1, 2**ly) if i % 2 == 1]

        self.basis_x = []
        self.basis_y = []

        for gx in self.gxs:
            for gy in self.gys:
                cx = self.getCX(gx)
                cy = self.getCY(gy)
                ax = ay = fun(cx, cy)
                if lx > 1:
                    ax -= parent.directParentX(gx).evalAt(cx)
                if ly > 1:
                    ay -= parent.directParentY(gy).evalAt(cy)

                self.basis_x.append(BasisFunction(gx, lx, ax))
                self.basis_y.append(BasisFunction(gy, ly, ay))

    def getCX(self, g):
        return (1/(2**ml)) * 2**(ml - self.lx) * g

    def getCY(self, g):
        return (1/(2**ml)) * 2**(ml - self.ly) * g

    def get_sum(self, xval, yval):
        b_2d = []
        for bx in self.basis_x:
            for by in self.basis_y:
                b_2d.append([bx.evalAt(x) * by.evalAt(y)
                             for x in xval
                             for y in yval])
        b_sum = b_2d[0]
        for b in b_2d[1:]:
            b_sum = np.add(b_sum, b)
        return b_sum

    def get_sum_x(self, xval):
        sum_x = [self.basis_x[0].evalAt(x) for x in xval]
        for bx in self.basis_x[1:]:
            sum_x = np.add(sum_x, [bx.evalAt(x) for x in xval])
        return sum_x

    def get_sum_y(self, yval):
        sum_y = [self.basis_y[0].evalAt(y) for y in yval]
        for by in self.basis_y[1:]:
            sum_y = np.add(sum_y, [by.evalAt(y) for y in yval])
        return sum_y

    def directParentX(self, g):
        cB = None
        cD = 1000
        for b in self.basis_x:
            if abs(b.gp - g) < cD:
                cD = abs(b.gp - g)
                cB = b
        return cB

    def directParentY(self, g):
        cB = None
        cD = 1000
        for b in self.basis_y:
            if abs(b.gp - g) < cD:
                cD = abs(b.gp - g)
                cB = b
        return cB

class BasisFunction:

    def __init__(self, gp, l, alpha):
        self.gp = gp
        self.l = l
        self.alpha = alpha

    def evalAt(self, x):
        return self.alpha * max(1 - abs(pow(2, self.l) * x - self.gp), 0)

SparseGrid()
