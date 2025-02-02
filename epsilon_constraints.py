from model import create_model, get_solution_dict
from gurobipy import GRB
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


def epsilon_constraints(num_zones: int, num_SRs: int, current_assignment: dict, distances: pd.DataFrame, index_values: pd.Series, objective_function: ObjectiveFunction, wl_interval: tuple[float, float] = (0.8, 1.2), epsilons_values : list[float] = [], epsilon_constraints: list[ObjectiveFunction] = [], center_bricks=[3, 13, 15, 21], pct:bool=False):
    status = GRB.OPTIMAL
    assignments = []  # Store the assignments
    scores = []  # Store the score for the solution found
    epsilons = [1000000 for _ in range(len(epsilon_constraints))]
    while status == GRB.OPTIMAL:
        print("epsilons: ", epsilons)
        model = create_model(num_zones=num_zones, num_SRs=num_SRs, current_assignment=current_assignment,
                             distances=distances, index_values=index_values, objective_function=objective_function, wl_interval=wl_interval,
                             epsilon=epsilons, epsilon_constraint=epsilon_constraints, pct=pct)
        model.params.outputflag = 0
        model.optimize()
        status = model.status
        print("current status: ", status)
        if status == GRB.OPTIMAL:
            #rebuilded_sol = np.array([[model.getVarByName(f"pct_zone_SR[{i},{j}]").X for j in range(num_SRs)] for i in range(num_zones)])
                
            assignments.append(get_solution_dict(
                model, num_zones, num_SRs, center_bricks))

            epsilons = []
            values = []
            for i, epsilon_func in enumerate(epsilon_constraints):
                '''current_vals.append(float(epsilon_func(
                    SR_matrix = rebuilded_sol, boolean_matrix = boolean_matrix, current_assignment, distances, index_values)))'''
                constraint = model.getConstrByName(f"epsilon_constraint_{i+1}")
                eps_func_value = model.getRow(constraint).getValue()
                values.append(eps_func_value)
                epsilons.append(eps_func_value - epsilons_values[i])

            scores.append((model.ObjVal, *values)) #TODO return all current vals

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
                                              distances=distances, index_values=index_values, objective_function=disruption, wl_interval=(0.8, 1.2), epsilons_values=[0.05, 0.01], epsilon_constraints=[distance, distance])

    print(scores)

    '''import matplotlib.pyplot as plt

    X = [el[0] for el in scores]
    y = [el[1] for el in scores]

    plt.plot(X, y, marker='o', color='b', linestyle=' ')
    plt.xlabel('Disruption')
    plt.ylabel('Distance')
    plt.show()'''

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Example 3D scores (replace with your data)

    # Extract X, Y, and Z coordinates
    X = [el[0] for el in scores]
    Y = [el[1] for el in scores]
    Z = [el[2] for el in scores]

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(X, Y, Z, marker='o', color='b')

    # Labels
    ax.set_xlabel('Disruption')
    ax.set_ylabel('Distance')
    ax.set_zlabel('Some Metric')  # Rename Z-axis as needed

    plt.show()

    '''assignments, scores = epsilon_constraints(num_zones=num_zones, num_SRs=num_SRs, current_assignment=current_assignment,
                                              distances=distances, index_values=index_values, objective_function=distance, wl_interval=(0.8, 1.2), epsilon_constraint=disruption)

    print(scores)

    import matplotlib.pyplot as plt

    X = [el[0] for el in scores]
    y = [el[1] for el in scores]

    plt.plot(X, y, marker='o', color='b', linestyle=' ')
    plt.xlabel('Distance')
    plt.ylabel('Disruption')
    plt.show()'''
