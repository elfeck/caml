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
        fig_height = fig_width*1        # height in inches
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
    f = newfig(0.2)

else:
    print("Show mode")
    import matplotlib.pyplot as plt
    #f = plt.figure()

def savefig(filename):
    plt.savefig('{}.pgf'.format(filename), bbox_inches="tight")
    plt.savefig('{}.pdf'.format(filename), bbox_inches="tight")
    plt.savefig('{}.png'.format(filename), bbox_inches="tight")

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

savedir = "../report/images/"

maxl = 2

def gridpoints(lx, ly):
    start = 0
    end = 2**maxl

    gx = [2**(maxl - lx) * i for i in range(1, 2**lx) if i % 2 == 1]
    gy = [2**(maxl - ly) * i for i in range(1, 2**ly) if i % 2 == 1]

    g = [(x, y) for x in gx for y in gy]

    return g

def drawBase(tp):
    ax = f.add_subplot(111)
    ax.axis([0, 2**maxl, 0, 2**maxl])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    pts = []
    for i in range(1, maxl+1):
        for j in range(1, maxl+1):
            if (i + j <= maxl + 1):
                pts += gridpoints(i, j)
    print(pts)
    for p in pts:
        color = "grey"
        if p == (3,2) and tp == 0:
            color = "red"
        ax.plot(p[0], p[1], "o", color=color)

    if tp >= 1:
        p = [3,2]
        modList = [[0.5, 0],
                   [-0.5, 0],
                   [0, 1],
                   [0, -1]]
        for m in modList:
            pp = np.add(p, m)
            color = "grey"
            if m == [0, -1] and tp == 1:
                color = "red"
            ax.plot(pp[0], pp[1], "o", color=color)

    if tp >= 2:
        p = [3, 1]
        modList = [[0.5, 0],
                   [-0.5, 0],
                   [0, 0.5],
                   [0, -0.5]]
        for m in modList:
            pp = np.add(p, m)
            color = "grey"
            ax.plot(pp[0], pp[1], "o", color=color)


        ax.plot([2,2], [0.5, 1.5], "bo")


drawBase(2)
savefig(savedir + "figure_4_2")
f.show()
