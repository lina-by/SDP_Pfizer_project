from gurobipy import Model, MVar, GRB, quicksum
import pandas as pd
from typing import Optional, Union

from objective_functions import distance, ObjectiveFunction
from model_new_sr import new_center_constraints, NewSRObjectiveFunction
from plots import plot_cities_attribution


def workload_constraints(model: Model, SR_matrix: MVar, index_values: pd.Series, wl_min: float = 0.8, wl_max: float = 1.2):
    num_zones, num_SRs = SR_matrix.shape
    for sr in range(num_SRs):
        # Add the constraint for the maximum workload
        model.addConstr(
            quicksum(SR_matrix[zone, sr] * index_values[zone]
                     for zone in range(num_zones)) <= wl_max,
            name=f"max_workload_sr_{sr}"
        )

        # Add the constraint for the minimum workload
        model.addConstr(
            quicksum(SR_matrix[zone, sr] * index_values[zone]
                     for zone in range(num_zones)) >= wl_min,
            name=f"min_workload_sr_{sr}"
        )


# assigner centre zone
def centre_assignment(model: Model, SR_matrix: MVar, center_brick: dict, seuil):
    for sr, center_zone in center_brick.items():
        model.addConstr(SR_matrix[center_zone, sr] >=
                        seuil, name=f"center_brick_{sr}")


def create_model(num_zones: int, num_SRs: int, current_assignment: dict, distances: pd.DataFrame,
                 index_values: pd.Series, objective_function: Union[ObjectiveFunction, NewSRObjectiveFunction],
                 wl_interval: tuple[float, float] = (0.8, 1.2), epsilon_constraint: Optional[ObjectiveFunction] = None,
                 epsilon: Optional[float] = None, seuil:float = 0.05, new_SRs:int=0, pct:bool=True):
    
    model = Model("pfizer_sr")
    center_brick = {sr: data["Center brick"]
                    for sr, data in current_assignment.items()}
    # pourcentage de la zone qui est occupée par le SR
    if pct:
        SR_matrix = model.addMVar((num_zones, num_SRs+new_SRs), lb=0, ub=1, vtype=GRB.CONTINUOUS, name="pct_zone_SR")
        # booléen la zone est occupée par le SR
        boolean_matrix = model.addMVar((num_zones, num_SRs+new_SRs), vtype=GRB.BINARY, name="zone_SR")     
        # Si le pct de la zone est non nul, le booléen est de 1
        model.addConstr(SR_matrix <= boolean_matrix, name="link_SR_boolean")
    else:
        SR_matrix = model.addMVar((num_zones, num_SRs+new_SRs), vtype=GRB.BINARY, name="pct_zone_SR")
        boolean_matrix = SR_matrix
    
    new_SRs_center = model.addMVar((num_zones, new_SRs), name='new_SRs')
    
    # chaque zone est occupée à 100%
    model.addConstr(SR_matrix.sum(axis=1) == 1, name="assign_one_SR")
    
    workload_constraints(model=model, SR_matrix=SR_matrix,
                         index_values=index_values, wl_min=wl_interval[0], wl_max=wl_interval[1])

    centre_assignment(model=model, SR_matrix=SR_matrix,
                      center_brick=center_brick, seuil=seuil)

    if new_SRs!=0:
        new_center_constraints(model, new_SRs_center, seuil)
        objective = objective_function(
            SR_matrix, boolean_matrix, new_SRs_center, current_assignment, distances, index_values)
    else:
        objective = objective_function(
        SR_matrix, boolean_matrix, current_assignment, distances, index_values)


    if epsilon is not None and epsilon_constraint is not None:
        constraint = epsilon_constraint(
            SR_matrix, boolean_matrix, current_assignment, distances, index_values)
        model.addConstr(constraint <= epsilon, name="epsilon_constraint")

    
    model.setObjective(objective, GRB.MINIMIZE)

    return model


def print_solution(model: Model, num_zones: int, num_SRs: int):
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found:\n")

        # Loop through the variables and access their values
        for sr in range(num_SRs):
            print(f"SR {sr+1} assignments:")
            for zone in range(num_zones):
                # Find the variable corresponding to zone and SR
                var_name = f"pct_zone_SR[{zone},{sr}]"
                var = model.getVarByName(var_name)
                if var.x > 0.5:  # If the value is 1, the zone is assigned to this SR
                    print(f"  Zone {zone+1} (Assigned)")
    else:
        print("No optimal solution found!")


def get_solution_dict(model: Model, num_zones: int, num_SRs: int, centers_list: list) -> dict:
    if model.status == GRB.OPTIMAL:
        assignment = {}

        for sr in range(num_SRs):
            assignment[sr] = {"Center brick": -1, "Assigned bricks": []}
            for zone in range(num_zones):
                var_name = f"pct_zone_SR[{zone},{sr}]"
                var = model.getVarByName(var_name)
                if var.x > 0:  # If the value is 1, the zone is assigned to this SR
                    assignment[sr]["Assigned bricks"].append((zone, var.x))
                    if zone in centers_list:
                        assignment[sr]["Center brick"] = zone

        return assignment
    else:
        return {}


if __name__ == '__main__':
    distances = pd.read_excel('data/distances.xlsx')
    distances = distances.drop(distances.columns[0], axis=1)
    distances.columns = range(len(distances.columns))

    index_values = pd.read_csv("data/bricks_index_values.csv")['index_value']

    num_zones = 22
    num_SRs = 4

    current_assignment = {
        0: {"Center brick": 3, "Assigned bricks": [3, 4, 5, 6, 7, 14]},
        1: {"Center brick": 13, "Assigned bricks": [9, 10, 11, 12, 13]},
        2: {"Center brick": 15, "Assigned bricks": [8, 15, 16, 17]},
        3: {"Center brick": 21, "Assigned bricks": [0, 1, 2, 18, 19, 20, 21]}
    }
    model = create_model(num_zones=num_zones, num_SRs=num_SRs, current_assignment=current_assignment,
                         distances=distances, objective_function=distance, index_values=index_values)
    model.optimize()
    print_solution(model, num_zones, num_SRs)
    plot_cities_attribution(
        get_solution_dict(
            model,
            num_zones,
            num_SRs,
            [current_assignment[i]["Center brick"]
                for i in current_assignment],
        )
    )
