from model import create_model, get_solution_dict, print_solution
from objective_functions import *

import pandas as pd
import numpy as np


def rebuild_sol(l: list, num_zones: int, num_SRs: int):
    '''
    Returns a matrix representing the solution from Gurobi
    '''
    mat = np.zeros((num_zones, num_SRs))

    for i in range(len(l)):
        if l[i]:
            row, col = i // num_SRs, i % num_SRs
            mat[row][col] = 1

    return mat


def epsilon_constraints(num_zones: int, num_SRs: int, current_assignment: dict, distances: pd.DataFrame, index_values: pd.Series, objective_function: ObjectiveFunction, wl_interval: tuple[float, float] = (0.8, 1.2), epsilon_constraint: Optional[ObjectiveFunction] = None, center_bricks=[3, 13, 15, 21]):
    status = GRB.OPTIMAL
    assignments = []  # Store the assignments
    scores = []  # Store the score for the solution found
    epsilon = None

    while status == GRB.OPTIMAL:
        model = create_model(num_zones=num_zones, num_SRs=num_SRs, current_assignment=current_assignment,
                             distances=distances, index_values=index_values, objective_function=objective_function, wl_interval=wl_interval,
                             epsilon=epsilon, epsilon_constraint=epsilon_constraint)
        model.params.outputflag = 0
        model.optimize()
        status = model.status

        if status == GRB.OPTIMAL:
            output_vars = [var.x for var in model.getVars()]
            rebuilded_sol = rebuild_sol(output_vars, num_zones, num_SRs)
            current_val = float(epsilon_constraint(
                rebuilded_sol, current_assignment, distances, index_values))

            assignments.append(get_solution_dict(
                model, num_zones, num_SRs, center_bricks))
            scores.append((model.ObjVal, current_val))

            if epsilon_constraint == distance:
                epsilon = current_val - 0.05
            else:
                epsilon = current_val - 0.02

    return assignments, scores


if __name__ == "__main__":
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
    assignments, scores = epsilon_constraints(num_zones=num_zones, num_SRs=num_SRs, current_assignment=current_assignment,
                                              distances=distances, index_values=index_values, objective_function=disruption, wl_interval=(0.8, 1.2), epsilon_constraint=distance)

    print(scores)

    import matplotlib.pyplot as plt

    X = [el[0] for el in scores]
    y = [el[1] for el in scores]

    plt.plot(X, y, marker='o', color='b', linestyle=' ')
    plt.xlabel('Disruption')
    plt.ylabel('Distance')
    plt.show()

    assignments, scores = epsilon_constraints(num_zones=num_zones, num_SRs=num_SRs, current_assignment=current_assignment,
                                              distances=distances, index_values=index_values, objective_function=distance, wl_interval=(0.8, 1.2), epsilon_constraint=disruption)

    print(scores)

    import matplotlib.pyplot as plt

    X = [el[0] for el in scores]
    y = [el[1] for el in scores]

    plt.plot(X, y, marker='o', color='b', linestyle=' ')
    plt.xlabel('Distance')
    plt.ylabel('Disruption')
    plt.show()
