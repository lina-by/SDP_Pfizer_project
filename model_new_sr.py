from gurobipy import Model, MVar
import numpy as np


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



def get_solution_dict_newSR(model: Model, num_zones: int, num_SRs: int, centers_list: list, new_SRs: int):
    """
    Extracts the assignment dictionary from the model and updates centers_list with new SR centers.
    """
    assignment_dict = {}
    SR_matrix = np.array([[model.getVarByName(f"pct_zone_SR[{i},{j}]").X for j in range(num_SRs + new_SRs)] for i in range(num_zones)])
    new_SRs_center = np.array([[model.getVarByName(f"new_SRs[{i},{j}]").X for j in range(new_SRs)] for i in range(num_zones)])
    
    for sr_id in range(num_SRs + new_SRs):
        assigned_bricks = [(i, SR_matrix[i, sr_id]) for i in range(num_zones) if SR_matrix[i, sr_id] > 0]
        center=None
        if sr_id<len(centers_list):
            center=centers_list[sr_id]
        assignment_dict[sr_id] = {"Center brick": center, "Assigned bricks": assigned_bricks}
    
    for j in range(new_SRs):
        center = np.argmax(new_SRs_center[:, j])
        assignment_dict[num_SRs + j]["Center brick"] = center
    
    return assignment_dict
  