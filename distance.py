from gurobipy import *
import pandas as pd

distances = pd.read_excel('Classeur1.xlsx', index_col=0)
index_values = pd.read_csv("bricks_index_values.csv").set_index('brick').squeeze()

# Create a new optimization model
model = Model("pfizer_sr")

zones = list(range(1, 23))
SR_values = [1, 2, 3, 4]
num_zones = len(zones)
num_SRs = len(SR_values)

current_assignment = {
    1: {"Center brick": 4, "Assigned bricks": [4, 5, 6, 7, 8, 15]},
    2: {"Center brick": 14, "Assigned bricks": [10, 11, 12, 13, 14]},
    3: {"Center brick": 16, "Assigned bricks": [9, 16, 17, 18]},
    4: {"Center brick": 22, "Assigned bricks": [1, 2, 3, 19, 20, 21, 22]}
}

center_brick = {sr: data["Center brick"] - 1 for sr, data in current_assignment.items()}


SR = model.addMVar((num_zones, num_SRs), vtype=GRB.BINARY, name="zone_SR")

# une zone par sr
model.addConstr(SR.sum(axis=1) == 1, name="assign_one_SR")

# charge de travail dans [0.8, 1.2]
for sr in SR_values:
    model.addConstr(
        quicksum(SR[zone-1, sr-1] * index_values[zone] for zone in zones) <= 1.2, 
        name=f"max_workload_sr_{sr}"
    )
    model.addConstr(
        quicksum(SR[zone-1, sr-1] * index_values[zone] for zone in zones) >= 0.8, 
        name=f"min_workload_sr_{sr}"
    )

# assigner centre zone
for sr, center_zone in center_brick.items():
    model.addConstr(SR[center_zone, sr - 1] == 1, name=f"center_brick_{sr}")



# obj
objective = 0
for zone in zones:
    for sr in SR_values:
        center_zone = center_brick[sr]
        objective += distances.loc[zone, center_zone] * SR[zone-1, sr - 1]

model.setObjective(objective, GRB.MINIMIZE)

model.optimize()

#print solution
if model.status == GRB.OPTIMAL:
    print("Optimal solution found:")
    
    # Print the assignment of zones to SRs
    for zone in zones:
        for sr in SR_values:
            if SR[zone-1, sr-1].x > 0.5:  # If zone is assigned to SR (binary variable)
                print(f"Zone {zone} is assigned to SR {sr}")

    # Print the value of the objective (total distance)
    print(f"Optimal total distance: {model.objVal}")
else:
    print("No optimal solution found.")