from model import create_model, get_solution_dict
from model_new_sr import get_solution_dict_newSR
from gurobipy import GRB
from objective_functions import *
import pandas as pd
import numpy as np
from plots import *

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


def get_values(model: Model):
    values = []

    for i in range(2):
        constraint = model.getConstrByName(f"epsilon_constraint_{i+1}")
        eps_func_value = model.getRow(constraint).getValue()
        values.append(eps_func_value)

    return (model.ObjVal, *values)

def epsilon_constraints(num_zones: int, num_SRs: int, current_assignment: dict, distances: pd.DataFrame, index_values: pd.Series, objective_function: ObjectiveFunction, new_SRs= 0, wl_interval: tuple[float, float] = (0.8, 1.2), epsilons_increment : list[float] = [], epsilon_constraints: list[ObjectiveFunction] = [], center_bricks=[3, 13, 15, 21], pct:bool=False):
    status = GRB.OPTIMAL
    assignments = []  # Store the assignments
    scores = []  # Store the score for the solution found
    epsilons = [num_SRs + new_SRs, wl_interval[1]]
    boule = True
    while status == GRB.OPTIMAL or boule: #epsilon 1
        print("New epsilon 1")
        boule = False
        

        status = GRB.OPTIMAL
        while status == GRB.OPTIMAL: #epsilon 2
            #print("new epsilon 2: ", epsilons)
            model = create_model(num_zones=num_zones, num_SRs=num_SRs, current_assignment=current_assignment,
                             distances=distances, index_values=index_values, objective_function=objective_function, wl_interval=wl_interval,
                             epsilon=epsilons, epsilon_constraint=epsilon_constraints, pct=pct, new_SRs=new_SRs)
            model.params.outputflag = 0
            model.optimize()
            status = model.status

            if status == GRB.OPTIMAL:
                if new_SRs:
                    new_assign = get_solution_dict_newSR(
                        model, num_zones, num_SRs, center_bricks, new_SRs)
                else:               
                    new_assign = get_solution_dict(
                        model, num_zones, num_SRs, center_bricks)
                #plot_cities_attribution(new_assign)
                assignments.append(new_assign)

                values = get_values(model)
                print(values)
                scores.append(values)
                epsilons[1] = values[2] - epsilons_increment[1]
                boule = True

        epsilons = [values[1] - epsilons_increment[0], 100000000]



    return assignments, scores


if __name__ == "__main__":
    distances = pd.read_excel('data/distances.xlsx')
    distances = distances.drop(distances.columns[0], axis=1)
    distances.columns = range(len(distances.columns))

    index_values = pd.read_csv("data/bricks_index_values.csv")['index_value']

    num_zones = 22
    num_SRs = 0

    current_assignment = {
        0: {"Center brick": 3, "Assigned bricks": [3, 4, 5, 6, 7, 14]},
        1: {"Center brick": 13, "Assigned bricks": [9, 10, 11, 12, 13]},
        2: {"Center brick": 15, "Assigned bricks": [8, 15, 16, 17]},
        3: {"Center brick": 21, "Assigned bricks": [0, 1, 2, 18, 19, 20, 21]}
    }
    assignments, scores = epsilon_constraints(num_zones=num_zones, num_SRs=num_SRs, current_assignment=current_assignment,
                                              distances=distances, index_values=index_values, objective_function=distance_newSR, wl_interval=(0.4, 1.6), epsilons_increment=[0.8, 0.05], epsilon_constraints=[disruption_newSR, min_max_newSR], new_SRs= 4)

    print(scores)

    '''import matplotlib.pyplot as plt

    X = [el[0] for el in scores]
    y = [el[1] for el in scores]

    plt.plot(X, y, marker='o', color='b', linestyle=' ')
    plt.xlabel('Disruption')
    plt.ylabel('Distance')
    plt.show()'''
    if scores != []:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Example 3D scores (replace with your data)

        # Extract X, Y, and Z coordinates
        X = [el[0] for el in scores]
        Y = [el[1] for el in scores]

        plt.plot(X, Y, marker='o', color='b', linestyle=' ')
        plt.xlabel('Distance')
        plt.ylabel('Disruption')
        plt.show()
        Z = [el[2] for el in scores]

        # Create 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot
        ax.scatter(X, Y, Z, marker='o', color='b')

        # Labels
        ax.set_xlabel('Total Distance')
        ax.set_ylabel('Disruption')
        ax.set_zlabel('Workload fairness')  # Rename Z-axis as needed

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
