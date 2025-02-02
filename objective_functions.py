from gurobipy import Model, MVar, quicksum
import pandas as pd
import inspect
from typing import Callable

ACCEPTED_KWARGS = {
    "model": Model,
    "SR_matrix": MVar,
    "boolean_matrix": MVar,
    "new_SRs_center": MVar,
    "current_assignment": dict,
    "distances": pd.DataFrame,
    "index_values": pd.Series,
}

class ObjectiveFunction(Callable):
    def __init__(self, func: Callable):
        self.func = func
        self.signature = inspect.signature(func)

        # Ensure function only accepts arguments from ACCEPTED_KWARGS
        for param in self.signature.parameters:
            if param not in ACCEPTED_KWARGS:
                raise ValueError(f"Invalid argument '{param}' in function '{func.__name__}'")

    def __call__(self, **kwargs):
        # Filter out arguments that are not in both the function signature and ACCEPTED_KWARGS
        valid_kwargs = {k: v for k, v in kwargs.items() if k in self.signature.parameters and k in ACCEPTED_KWARGS}
        return self.func(**valid_kwargs)

@ObjectiveFunction
def disruption(SR_matrix:MVar, current_assignment: dict, index_values:pd.Series):
    num_zones, num_SRs = SR_matrix.shape
    objective = 0
    for sr in range(num_SRs):
        for former_bricks in current_assignment[sr]["Assigned bricks"]:
            objective += index_values[former_bricks] * (1-SR_matrix[former_bricks, sr])

    return objective

@ObjectiveFunction
def distance(boolean_matrix:MVar, current_assignment: dict, distances:pd.DataFrame):
    num_zones, num_SRs = boolean_matrix.shape
    objective = 0
    center_brick = {sr: data["Center brick"] for sr, data in current_assignment.items()}
    for zone in range(num_zones):
        for sr in range(num_SRs):
            center_zone = center_brick[sr]
            objective += distances.loc[zone, center_zone] * boolean_matrix[zone, sr]

    return objective


@ObjectiveFunction
def distance_newSR(boolean_matrix:MVar, new_SRs_center:MVar, current_assignment: dict, distances:pd.DataFrame):
    num_zones, num_SRs = boolean_matrix.shape
    num_zones, num_new_SRs = new_SRs_center.shape
    nb_former_SR = num_SRs-num_new_SRs
    objective = 0
    center_brick = {sr: data["Center brick"] for sr, data in current_assignment.items()}
    for zone in range(num_zones):
        for sr in range(nb_former_SR):
            center_zone = center_brick[sr]
            objective += distances.loc[zone, center_zone] * boolean_matrix[zone, sr]
        for new_SR in range(num_new_SRs):
            for center_zone in range(num_zones):
                objective += distances.loc[zone, center_zone] * boolean_matrix[zone, nb_former_SR+new_SR] *\
                            new_SRs_center[center_zone, new_SR]

    return objective


@ObjectiveFunction
def disruption_newSR(new_SRs_center:MVar, current_assignment: dict):
    num_zones, num_SRs = new_SRs_center.shape
    objective = 0
    for sr, data in current_assignment.items():
        center_brick=data["Center brick"]
        objective += new_SRs_center[center_brick, sr] - 1
    return objective

@ObjectiveFunction
def min_max_newSR(model:Model, SR_matrix:MVar, index_values:pd.Series):
    maximum = model.addVar()
    num_zones, num_SRs = SR_matrix.shape
    for sr in range(num_SRs):
        # Add the constraint for the maximum workload
        model.addConstr(
            quicksum(SR_matrix[zone, sr] * index_values[zone]
                     for zone in range(num_zones)) <= maximum,
            name=f"max_workload_sr_{sr}"
        )
    return maximum