from gurobipy import *
import numpy as np
import csv

starting_assignment = {
    1: {"Center brick": 4, "Assigned bricks": [4, 5, 6, 7, 8, 15]},
    2: {"Center brick": 14, "Assigned bricks": [10, 11, 12, 13, 14]},
    3: {"Center brick": 16, "Assigned bricks": [9, 16, 17, 18]},
    4: {"Center brick": 22, "Assigned bricks": [1, 2, 3, 19, 20, 21, 22]}
}

starting_assignment_mat = np.zeros((22, 4), dtype=np.int64)
for sr in range(4):
    for assigned_brick in starting_assignment[sr+1]["Assigned bricks"]:
        starting_assignment_mat[assigned_brick-1][sr] = 1


def read_brick_values(path: str = 'data/bricks_index_values.csv'):
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        index_values = {}
        for line in reader:
            if line[0] != 'brick':
                index_values[int(line[0])] = float(line[1])
        return index_values


# , starting_assignment_mat):
def disruption_objective(disruption_vars, nb_bricks, nb_sr):
    score = 0
    for brick in range(nb_bricks):
        for sr in range(nb_sr):
            score += disruption_vars[brick][sr]
    return score


def disruption_model(interval_lb, interval_ub, nb_bricks, nb_sr):
    brick_values = read_brick_values()
    model = Model("Disruption model")
    vars = model.addMVar((len(brick_values), nb_sr), vtype=GRB.BINARY)

    disruption_vars = model.addMVar(
        (len(brick_values), nb_sr), vtype=GRB.BINARY)

    for brick in range(nb_bricks):
        for sr in range(nb_sr):
            model.addConstr(disruption_vars[brick][sr] >= (
                vars[brick][sr] - starting_assignment_mat[brick][sr]))
            model.addConstr(disruption_vars[brick][sr] >= (
                starting_assignment_mat[brick][sr] - vars[brick][sr]))

    model.update()

    for sr in range(nb_sr):
        model.addConstr(
            quicksum(vars[brick, sr-1] * brick_values[brick+1]
                     for brick in range(nb_bricks)) <= 1.2,
            name=f"max_workload_sr_{sr}"
        )
    model.addConstr(
        quicksum(vars[brick, sr-1] * brick_values[brick+1]
                 for brick in range(nb_bricks)) >= 0.8,
        name=f"min_workload_sr_{sr}"
    )

    model.setObjective(disruption_objective(
        disruption_vars, nb_bricks, nb_sr), GRB.MINIMIZE)

    model.optimize()

    if model.status == GRB.OPTIMAL:
        print("Optimal solution found:")

        # Print the assignment of zones to SRs
        zones = list(range(1, 23))
        SR_values = [1, 2, 3, 4]
        for zone in zones:
            for sr in SR_values:
                # If zone is assigned to SR (binary variable)
                if vars[zone-1, sr-1].x > 0.5:
                    print(f"Zone {zone} is assigned to SR {sr}")

        # Print the value of the objective (total distance)
        print(f"Optimal total distance: {model.objVal}")
    else:
        print("No optimal solution found.")


if __name__ == "__main__":
    print(read_brick_values())
    disruption_model(0.8, 1.2, 22, 4)
