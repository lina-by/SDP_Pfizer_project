from gurobipy import Model, MVar
import pandas as pd
from typing import Callable


class NewSRObjectiveFunction(Callable):
    def __call__(self, SR_matrix: MVar, boolean_matrix:MVar, new_SRs_center:MVar,
                current_assignment: dict, distances: pd.DataFrame, index_values:pd.Series):
        pass


def new_center_constraints(model:Model, SR_matrix:MVar, new_SRs_center:MVar, seuil:float):
    num_zones, new_SRs = new_SRs_center.shape 
    # Un centre par SR
    model.addConstr(new_SRs_center.sum(axis=0) == 1, name="assign_one_center")
    # workload du centre au dessus du seuil
    for new_SR in range(new_SRs):
        workload_in_SR_center = 0
        for zone in range(num_zones):
            workload_in_SR_center += SR_matrix[zone, new_SR] * new_SRs_center[zone, new_SR]
        model.addConstr(workload_in_SR_center >=
                        seuil, name=f"center_brick_new_{new_SR}")
        
def distance(SR_matrix:MVar, boolean_matrix:MVar, new_SRs_center:MVar, current_assignment: dict, distances:pd.DataFrame, index_values:pd.Series):
    num_zones, num_SRs = SR_matrix.shape
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

       
def disruption_center(SR_matrix:MVar, boolean_matrix:MVar, new_SRs_center:MVar, current_assignment: dict, distances:pd.DataFrame, index_values:pd.Series):
    num_zones, num_SRs = SR_matrix.shape
    objective = 0
    for sr in range(num_SRs):
        for former_bricks in current_assignment[sr]["Assigned bricks"]:
            objective += index_values[former_bricks] * (1-SR_matrix[former_bricks, sr])

    return objective