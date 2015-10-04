import numpy as np
import math
import matplotlib.pyplot as plt

savedir = "../slides/images/"

def fun(x):
    return 1 + math.sin(math.pi * 1.45 * x)

step = 0.001
start = 0
end = 1

xval = np.arange(start, end + step, step)
yval = [fun(x) for x in xval]

gridpoints = [i for i in range(0, 8)]

def singleBasis1():
    f = plt.figure()
    sb = f.add_subplot(111)
    sb.set_xticklabels([])
    plt.axis([start, end, 0, 2.2])
    plt.xticks([])

    plt.plot(xval, yval, c="red")
    plt.savefig(savedir + "singlebasis_1.png", bbox_inches="tight")
    plt.show()


def singleBasis2():
    f = plt.figure()
    sb = f.add_subplot(111)
    sb.set_xticklabels([i for i in range(1, len(gridpoints) + 1)])
    #sb.set_yticklabels([])
    plt.axis([start, end, 0, 2.2])
    plt.xticks([(1/8.0) * g for g in gridpoints])

    # function
    plt.plot(xval, yval, c="red")

    rectpoints = [(1/8) * i for i in range(0, 8)]
    for r in rectpoints:
        plt.plot((r, r), (0, 2.2), color="black")

    plt.savefig(savedir + "singlebasis_2.png", bbox_inches="tight")
    plt.show()

def singleBasis3():
    f = plt.figure()
    sb = f.add_subplot(111)
    sb.set_xticklabels([i for i in range(1, len(gridpoints) + 1)])
    #sb.set_yticklabels([])
    plt.axis([start, end, 0, 2.2])
    plt.xticks([(1/8.0) * g for g in gridpoints])
    plt.gca().get_xticklabels()[2].set_color("blue")

    # function
    plt.plot(xval, yval, color="red")

    # base nr 2
    point = (1/8.0) * gridpoints[2]
    alpha = fun(point)
    ys = [hatfun(x, 3, gridpoints[2]) for x in xval]
    plt.plot(xval, ys, c="blue", linestyle="solid")

    t3 = plt.text(point + 0.12, alpha - 1.6, r'$\phi_3(x)$', color="blue")
    t3.set_fontsize(16)

    plt.savefig(savedir + "singlebasis_3.png", bbox_inches="tight")
    plt.show()

def singleBasis4():
    f = plt.figure()
    sb = f.add_subplot(111)
    sb.set_xticklabels([i for i in range(1, len(gridpoints) + 1)])
    #sb.set_yticklabels([])
    plt.axis([start, end, 0, 2.2])
    plt.xticks([(1/8.0) * g for g in gridpoints])
    plt.gca().get_xticklabels()[2].set_color("blue")

    # function
    plt.plot(xval, yval, color="red")

    # base nr 2
    point = (1/8.0) * gridpoints[2]
    alpha = fun(point)
    ys = [hatfun(x, 3, gridpoints[2]) for x in xval]
    ys_a = [alpha * hatfun(x, 3, gridpoints[2]) for x in xval]
    plt.plot(xval, ys_a, color="blue")
    plt.plot(xval, ys, color="grey", linestyle="dotted")

    # alpha line
    plt.plot((point, point), (0, alpha), color="blue", linestyle="dashed")

    # labels
    t1 = plt.text(point + 0.01, alpha - 1.6, r'$\alpha_3$', color="blue")
    t1.set_fontsize(12)

    t2 = plt.text(point + 0.12, alpha - 1.6,
                  r'$\alpha_3 \cdot \phi_3(x)$', color="blue")
    t2.set_fontsize(16)

    t3 = plt.text(point + 0.117, alpha -1.5, r'$\phi_3(x)$', color="grey")
    t3.set_fontsize(12)

    plt.savefig(savedir + "singlebasis_4.png", bbox_inches="tight")
    plt.show()

def singleBasis5():
    f = plt.figure()
    sb = f.add_subplot(111)
    sb.set_xticklabels([i for i in range(1, len(gridpoints) + 1)])
    #sb.set_yticklabels([])
    plt.axis([start, end, 0, 2.2])
    plt.xticks([(1/8.0) * g for g in gridpoints])

    # function
    plt.plot(xval, yval, color="red")

    # basis fun
    ys_a = []
    for g in gridpoints:
        point = (1/8.0) * g
        alpha = fun(point)
        ys_a.append([alpha * hatfun(x, 3, g) for x in xval])
        plt.plot(xval, ys_a[-1], color="blue", linestyle="dotted")

    #sum
    ys_sum = [0 for i in ys_a[0]]
    for ys_i in ys_a:
        for i in range(len(ys_i)):
            ys_sum[i] += ys_i[i]
    plt.plot(xval, ys_sum, color="blue", linewidth=2)

    t1 = plt.text(0.8, 0.7,
                  r'$\sum_{\{1..8\}}^{i}{\alpha_i \cdot \phi_i}$', color="blue")
    t1.set_fontsize(16)

    plt.savefig(savedir + "singlebasis_5.png", bbox_inches="tight")
    plt.show()

def hatfun(x, l = 0, i = 0):
    return max(1 - abs(pow(2, l) * x - i), 0)
