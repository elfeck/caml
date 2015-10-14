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
        fig_height = fig_width*0.6
        # height in inches
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
    f = newfig(0.6)

else:
    print("Show mode")
    import matplotlib.pyplot as plt
    f = plt.figure()

def savefig(filename):
    #plt.savefig('{}.pgf'.format(filename), bbox_inches="tight")
    plt.savefig('{}.pdf'.format(filename), bbox_inches="tight")
    plt.savefig('{}.png'.format(filename), bbox_inches="tight")

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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

savedir = "../report/images/"

start = 0
end = 1
step = 0.005
maxl = 3

xval = yval = np.arange(start, end + step, step)
xmesh, ymesh = np.meshgrid(xval, yval)

def gridpoints(lx, ly):
    start = 0
    end = 2**maxl

    gx = [i for i in range(1, 2**lx) if i % 2 == 1]
    gy = [i for i in range(1, 2**ly) if i % 2 == 1]

    g = [(x, y) for x in gx for y in gy]

    return list(zip(*g))

def hatfun(x, l = 0, i = 0):
    return max(1 - abs(pow(2, l) * x - i), 0)

def get2dHats(lx, ly):
    pts = gridpoints(lx, ly)
    hatx = []
    haty = []
    for g in pts[0]:
        hatx.append([hatfun(x, lx, g) for x in xval])
    for g in pts[1]:
        haty.append([hatfun(y, ly, g) for y in yval])

    zs = [[x * y for x in cx for y in cy] for cx in hatx for cy in haty]
    zval = np.zeros(len(zs[0]))
    for z in zs:
        zval = zval + z

    zmesh = np.reshape(np.array(zval), xmesh.shape)
    return zmesh


def drawAll():
    k = 1
    maxl = 3
    print("start")
    for i in range(1, maxl+1):
        for j in range(1, maxl+1):
            zmesh = get2dHats(i, j)
            ax = f.add_subplot(maxl, maxl, k, projection="3d")
            k+=1
            cmapp = cm.coolwarm if i + j <= 4 else cm.Greys
            ax.plot_surface(xmesh, ymesh, zmesh, cmap=cmapp, linewidth=0,
                            cstride=4,rstride=4)
            #ax.axis("off")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
    f.subplots_adjust(hspace=0, wspace=0)

drawAll()
savefig(savedir + "figure_2")
plt.show()
