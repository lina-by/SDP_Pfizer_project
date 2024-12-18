from gurobipy import *
import pandas as pd
from typing import Callable, Optional

class ObjectiveFunction(Callable):
    def __call__(self, SR_matrix: MVar, current_assignment: dict, distances: pd.DataFrame, index_values:pd.Series):
        pass

    
def disruption(SR_matrix:MVar, current_assignment: dict, distances:pd.DataFrame, index_values:pd.Series):
    num_zones, num_SRs = SR_matrix.shape
    objective = 0
    for sr in range(num_SRs):
        for former_bricks in current_assignment[sr]["Assigned bricks"]:
            objective += index_values[former_bricks] * (1-SR_matrix[former_bricks, sr])

    return objective

def distance(SR_matrix:MVar, current_assignment: dict, distances:pd.DataFrame, index_values:pd.Series):
    num_zones, num_SRs = SR_matrix.shape
    objective = 0
    center_brick = {sr: data["Center brick"] for sr, data in current_assignment.items()}
    for zone in range(num_zones):
        for sr in range(num_SRs):
            center_zone = center_brick[sr]
            objective += distances.loc[zone, center_zone] * SR_matrix[zone, sr]

    return objective

