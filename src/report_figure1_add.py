show_mode = False

if show_mode is False:
    print("Latex mode")
    import numpy as np
    import matplotlib as mpl
    mpl.use('pgf')
    def figsize(scale):
        fig_width_pt = 516.000           # Get this from LaTeX using \the\textwidth
        inches_per_pt = 1.0/72.27        # Convert pt to inch
        golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio
        fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
        fig_height = fig_width*0.6           # height in inches
        fig_size = [fig_width,fig_height]
        return fig_size

    pgf_with_latex = {    # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
        "text.usetex": True,                # use LaTeX to write all text
        "font.family": "serif",
        # blank entries should cause plots to inherit fonts from the document
        "font.serif": [],
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": 10,    # LaTeX default is 10pt font.
        "text.fontsize": 10,
        "legend.fontsize": 8,    # Make the legend/label fonts a little smaller
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.figsize": figsize(0.9), # default fig size of 0.9 textwidth
        "pgf.preamble": [
            # use utf8 fonts becasue your computer can handle it :)
            r"\usepackage[utf8x]{inputenc}",
            # plots will be generated using this preamble
            r"\usepackage[T1]{fontenc}",
        ]
    }
    mpl.rcParams.update(pgf_with_latex)

    import matplotlib.pyplot as plt

    def newfig(width):
        plt.clf()
        fig = plt.figure(figsize=figsize(width))
        return fig
    f = newfig(0.33)

else:
    print("Show mode")
    import matplotlib.pyplot as plt
    f = plt.figure()

def savefig(filename):
    plt.savefig('{}.pgf'.format(filename), bbox_inches="tight")
    plt.savefig('{}.png'.format(filename), bbox_inches="tight")
    plt.savefig('{}.pdf'.format(filename), bbox_inches="tight")

#
#
#
# CODE FROM HERE
#
#
#

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

savedir = "../report/images/"

def fun(x):
    return (3/(x + 1)**4) * math.sin(math.pi * x)


def hatfun(x, l = 0, i = 0):
    return max(1 - abs(pow(2, l) * x - i), 0)

class SparseGrid:

    def __init__(self):
        subs = [Subspace(1, None)]
        subs.append(Subspace(2, subs[0]))
        subs.append(Subspace(3, subs[1]))
        #subs.append(Subspace(4, subs[2]))

        xval = np.arange(start, end + step, step)

        ax = f.add_subplot(111)
        ax.set_xticks([i/2**3 for i in range(1,8)])
        ax.set_xticklabels([
            "(3,1)", "(2,1)", "(3,3)", "(1,1)",
            "(3,5)", "(2,3)", "(3,7)"])
        ax.axis([0, 1, -0.1, 1])
        plt.axhline(0, color="black")
        #ax.spines['bottom'].set_position('zero')
        #ax.set_yticklabels([])

        y_sum = []
        for s in subs:
            if s.lx == 1:
                col = "brown"
            elif s.lx == 2:
                col = "orange"
            else:
                col = "green"
            if len(y_sum) == 0:
                y_sum = s.get_sum(xval)
            else:
                y_sum = np.add(y_sum, s.get_sum(xval))
            ax.plot(xval, s.get_sum(xval), color=col, linestyle="solid")
            #axs.axis([0, 1, -1, 1])
        ax.plot(xval, y_sum)
        #ax.plot(xval, [fun(x) for x in xval], color="red")

        #ax.plot(xval, [-0.199 for x in xval], color="black")

class Subspace:

    def __init__(self, lx, parent):
        self.lx = lx
        self.gxs = [i for i in range(1, 2**lx) if i % 2 == 1]

        self.basis_x = []

        for gx in self.gxs:
            cx = self.getCX(gx)

            bPar = None
            if parent is not None:
                bPar = parent.directParentX(cx)
            self.basis_x.append(BasisFunction(gx, cx, lx, fun(cx), bPar))

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

    def __init__(self, gp, gc, l, feval, parent):
        self.gp = gp # gridpoint in level
        self.gc = gc # girdpoint coord overall
        self.l = l
        self.parent = parent
        self.feval = feval
        self.alpha = feval

        p = parent
        while p is not None:
            self.alpha -= p.evalAt(gc)
            p = p.parent

    def evalAt(self, x):
        return self.alpha * max(1 - abs(pow(2, self.l) * x - self.gp), 0)

SparseGrid()
savefig(savedir + "figure_1_3")
plt.show()
