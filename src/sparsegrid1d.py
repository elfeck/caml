import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

start = 0
end = 1
step = 0.001
ml = 3

savedir = "../slides/images/"

def fun(x):
    return math.sin(2 * math.pi * x)


def hatfun(x, l = 0, i = 0):
    return max(1 - abs(pow(2, l) * x - i), 0)

class SparseGrid:

    def __init__(self):
        f = plt.figure()
        subs = [Subspace(1, None)]
        subs.append(Subspace(2, subs[0]))
        subs.append(Subspace(3, subs[1]))

        xval = np.arange(start, end + step, step)

        gs = gridspec.GridSpec(3, 2)
        axs = []
        axs.append(plt.subplot(gs[0, 0]))
        axs.append(plt.subplot(gs[1, 0]))
        axs.append(plt.subplot(gs[2, 0]))
        axs.append(plt.subplot(gs[0, 1]))
        axs.append(plt.subplot(gs[1, 1]))
        axs.append(plt.subplot(gs[2, 1]))

        for ax in axs:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        y_sum = []
        k = 0
        for s in subs:
            if len(y_sum) == 0:
                y_sum = s.get_sum(xval)
            else:
                y_sum = np.add(y_sum, s.get_sum(xval))
            axs[k].plot(xval, s.get_sum(xval))
            axs[k].axis([0, 1, -1, 1])
            t = axs[k].text(0.02, 0.7, "l=" + str(k+ 1))
            t.set_fontsize(14)
            k += 1
        axs[-2].plot(xval, y_sum)
        axs[-2].text(0.8, 0.7, "hirac")
        axs[-1].plot(xval, [fun(x) for x in xval])

        #full
        ys_a = []
        gridpoints = range(1,8)
        for g in gridpoints:
            point = (1/8.0) * g
            alpha = fun(point)
            ys_a.append([alpha * hatfun(x, 3, g) for x in xval])

        ys_sum = ys_a[0]
        for ys_i in ys_a[1:]:
            ys_sum = np.add(ys_sum, ys_i)
        #\full
        axs[-3].plot(xval, ys_sum, color="grey")
        axs[-3].text(0.8, 0.7, "nodal")

        plt.savefig(savedir + "sparsegrid_1d.png", bbox_inches="tight")
        plt.show()

class Subspace:

    def __init__(self, lx, parent):
        self.lx = lx
        self.gxs = [i for i in range(1, 2**lx) if i % 2 == 1]
        self.parent = parent

        self.basis_x = []

        for gx in self.gxs:
            cx = self.getCX(gx)
            ax = fun(cx)
            hatSub = 0
            p = parent
            #if p != None:
            while p is not None:
                hatSub += p.directParentX(cx).evalAt(cx)
                print(str(gx) + " (" + str(lx) + ") : " +
                      str(p.directParentX(cx).evalAt(cx)))
                p = p.parent
            ax -= hatSub

            self.basis_x.append(BasisFunction(gx, lx, ax))

    def getCX(self, g):
        return (1/(2**ml)) * 2**(ml - self.lx) * g

    def get_sum(self, xval):
        b_sum = [self.basis_x[0].evalAt(x) for x in xval]
        for b in self.basis_x[1:]:
            b_sum = np.add(b_sum, [b.evalAt(x) for x in xval])
        return b_sum

    def directParentX(self, g):
        cB = None
        cD = 1000
        for b in self.basis_x:
            if abs(self.getCX(b.gp) - g) < cD:
                cD = abs(self.getCX(b.gp) - g)
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
