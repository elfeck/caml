import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

savedir = "../slides/images/"

def hatfun(x, i):
    return max(1 - abs(8 * x - i), 0)

start = 0
end = 1
step = 0.025

xval = yval = np.arange(start, end + step, step)
xmesh, ymesh = np.meshgrid(xval, yval)

zval = [hatfun(x, 4) * hatfun(y, 3) for x in xval for y in yval]
zmesh = np.reshape(zval, xmesh.shape)


f = plt.figure()
ax = f.add_subplot(111, projection="3d")

ax.plot(xval, np.ones(len(xval)), [hatfun(x, 3) for x in xval], color="blue")
ax.plot(np.zeros(len(xval)), xval, [hatfun(x, 4) for x in xval], color="blue")
ax.plot_surface(xmesh, ymesh, zmesh, cstride = 5, rstride = 5)

plt.xticks([i/8 for i in range(1, 8)])
plt.yticks([i/8 for i in range(1, 8)])
ax.set_yticklabels(["1","2","3","4","5","6","7"])
ax.set_xticklabels(["7", "6", "5", "4", "3", "2", "1"])
#ax.set_zticklabels([])


plt.savefig(savedir + "tensor.png", bbox_inches="tight")
plt.show()
