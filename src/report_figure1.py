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


import matplotlib.pyplot as plt
import numpy as np
import math

savedir = "../report/images/"

def fun(x):
    return math.sin(math.pi * 2 * x)

def fun2(x):
    return (3/(x + 1)**4) * math.sin(math.pi * x)

step = 0.001
start = 0
end = 1
upper = 1
lower = 0

xval = np.arange(start, end + step, step)
yval = [fun(x) for x in xval]
yval2 = [fun2(x) for x in xval]

gridpoints = [i for i in range(1, 8)]

def plot_reportfig_1():
    sb = f.add_subplot(111)
    sb.set_xticklabels([i for i in range(1, len(gridpoints) + 1)])
    #sb.set_yticklabels([])
    plt.axis([start, end, lower, upper])
    plt.xticks([(1/8.0) * g for g in gridpoints])

    # function
    plt.plot(xval, yval2, color="red")

    #lines
    #rectpoints = [(1/8) * i for i in range(1, 8)]
    #for r in rectpoints:
        #plt.plot((r, r), (-1, 1), color="black")

    for g in gridpoints:
        point = (1/8.0) * g
        alpha = fun(point)
        ys = [hatfun(x, 3, g) for x in xval]
        ys_a = [alpha * hatfun(x, 3, g) for x in xval]
        c = "blue" if g % 2 == 0 else "red"
        sb.axvline(point, color="grey")
        #sb.plot(xval, ys, color="grey", linestyle="solid")

def plot_reportfig_2():
    sb2 = f.add_subplot(111)
    sb2.set_xticklabels([i for i in range(1, len(gridpoints) + 1)])
    #sb.set_yticklabels([])
    plt.axis([start, end, lower, upper])
    plt.xticks([(1/8.0) * g for g in gridpoints])
    #sb2.plot(xval, yval2, color="red")

    ys_a = []
    for g in gridpoints:
        point = (1/8.0) * g
        alpha = fun2(point)
        ys_a.append([alpha * hatfun(x, 3, g) for x in xval])
        plt.plot(xval, ys_a[-1], color="blue", linestyle="dotted")

    ys_sum = [0 for i in ys_a[0]]
    for ys_i in ys_a:
        for i in range(len(ys_i)):
            ys_sum[i] += ys_i[i]
    plt.plot(xval, ys_sum, color="blue")

    #f.set_size_inches(800/mydpi, 600/mydpi, dpi = mydpi)

def hatfun(x, l = 0, i = 0):
    return max(1 - abs(pow(2, l) * x - i), 0)

plot_reportfig_1()
savefig(savedir + "figure_1_1")
f.show()
