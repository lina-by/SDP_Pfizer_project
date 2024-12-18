import numpy as np
from gurobipy import *

brick_distance = np.genfromtxt(
    "data/brick_rp_distances.csv", delimiter=",", skip_header=1
)
index_values = np.genfromtxt(
    "data/bricks_index_values.csv", delimiter=",", skip_header=1
)
