import numpy as np
import matplotlib.pyplot as plt

def points_show(points):
    points = np.array(points).T
    plt.scatter(points[0], points[1])
    plt.xlim(-1,256)
    plt.ylim(-1,256)
    plt.show()
