# Modules de base
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# Module relatif à Gurobi
from gurobipy import Model, GRB


def rank(
    list_alternatives: np.ndarray,
    partial_ranking: list,
    L: int,
    minmax: dict,
    eps: float,
):
    """
    This function creates the preference score model to rank the alternatives regarding the partial ranking given as input.

    Inputs:
        list_alternatives:  array of size (number of alternatives, number of criteria) representing the score on each criteria and for each alternative
        partial ranking:    list of pairs of alternatives exhibiting a preference. For example, the pair [0,1] means alternative 0 is scored higher than alternaative 1
        L:                  number of pieces for the piecewize-affine function of each criteria
        minmax:             dictionnary whose keys is the id of the criteria and the value is a pair [minimum value, maximum value] for the given criteria.
        eps:                epsilon value determining strictly inequality constraints (a > b     <=>    a >= b + eps)
    """
    model = Model("preferences")

    num_preferences = list_alternatives.shape[0]
    num_criteria = len(minmax)

    turning_point_criteria = np.zeros((num_criteria, L + 1))
    for id, criteria in enumerate(minmax):
        min, max = minmax[criteria]
        for turning_point in range(L + 1):
            turning_point_criteria[id, turning_point] = (
                min + turning_point * (max - min) / L
            )

    segment_identification = {}
    for [id0, id1] in partial_ranking:
        if id0 not in segment_identification:
            segment_identification[id0] = []
            values = list_alternatives[id0]
            for i, value in enumerate(values):
                min, max = minmax[i]
                k = math.floor(L * (value - min) / (max - min))
                xik = min + k * (max - min) / L
                segment_identification[id0].append([k, L * (value - xik) / (max - min)])

        if id1 not in segment_identification:
            segment_identification[id1] = []
            values = list_alternatives[id1]
            for i, value in enumerate(values):
                min, max = minmax[i]
                k = math.floor(L * (value - min) / (max - min))
                xik = min + k * (max - min) / L
                segment_identification[id1].append([k, L * (value - xik) / (max - min)])

    sik_matrix = model.addMVar(
        (num_criteria, L + 1), vtype=GRB.CONTINUOUS, name="Sik", lb=0, ub=1
    )

    sigmaplus = model.addMVar(
        (num_preferences), vtype=GRB.CONTINUOUS, name="sigmaplus", lb=0
    )

    sigmamoins = model.addMVar(
        (num_preferences), vtype=GRB.CONTINUOUS, name="sigmamoins", lb=0
    )

    model.addConstr(sik_matrix[:, -1].sum() == 1, name="sum of max = 1")

    for i in range(num_criteria):
        model.addConstr(sik_matrix[i][0] == 0, name=f"initialisation_to_0_{i}")

        for k in range(L):
            model.addConstr(
                sik_matrix[i][k] <= sik_matrix[i][k + 1], name=f"monotonie_{i}_{k}"
            )

    for [id0, id1] in partial_ranking:
        six0 = 0
        six1 = 0

        for i in range(num_criteria):
            if segment_identification[id0][i][0] == L:
                six0 += sik_matrix[i, -1]
            else:
                six0 += (
                    segment_identification[id0][i][1]
                    * (
                        sik_matrix[i, segment_identification[id0][i][0] + 1]
                        - sik_matrix[i, segment_identification[id0][i][0]]
                    )
                    + sik_matrix[i, segment_identification[id0][i][0]]
                )

            if segment_identification[id1][i][0] == L:
                six1 += sik_matrix[i, -1]
            else:
                six1 += (
                    segment_identification[id1][i][1]
                    * (
                        sik_matrix[i, segment_identification[id1][i][0] + 1]
                        - sik_matrix[i, segment_identification[id1][i][0]]
                    )
                    + sik_matrix[i, segment_identification[id1][i][0]]
                )

        model.addConstr(
            six0 + sigmaplus[id0] - sigmamoins[id0]
            >= six1 + sigmaplus[id1] - sigmamoins[id1] + eps
        )

    model.setObjective(sigmaplus.sum() + sigmamoins.sum(), GRB.MINIMIZE)

    return model, sik_matrix, turning_point_criteria


def plot_score_curves(stop_points, score_matrix_result, hospitals):
    plt.figure(figsize=(15, 5))
    criteria_labels = [f"Critère {i + 1}" for i in range(stop_points.shape[1])]

    for criterion in range(stop_points.shape[1]):
        plt.subplot(stop_points.shape[1] // 3, 3, criterion + 1)

        # Tracer la courbe de variation du score
        plt.plot(
            stop_points[:, criterion],
            score_matrix_result[:, criterion],
            marker="o",
            label=f"Score {criteria_labels[criterion]}",
        )

        # Ajouter les universités sur la courbe avec interpolation
        for hosp_idx, hosp in enumerate(hospitals):
            l = np.searchsorted(stop_points[:, criterion], hosp[criterion])
            if l == 0:
                interpolated_score = score_matrix_result[0, criterion]
            elif l >= len(stop_points):
                interpolated_score = score_matrix_result[-1, criterion]
            else:
                dx = (hosp[criterion] - stop_points[l - 1, criterion]) / (
                    stop_points[l, criterion] - stop_points[l - 1, criterion]
                )
                dy = dx * (
                    score_matrix_result[l, criterion]
                    - score_matrix_result[l - 1, criterion]
                )
                interpolated_score = score_matrix_result[l - 1, criterion] + dy

            plt.scatter(hosp[criterion], interpolated_score, marker="x", color="red")
            plt.text(
                hosp[criterion],
                interpolated_score,
                f"Hospital {hosp_idx}",
                fontsize=9,
                verticalalignment="bottom",
            )

        plt.xlabel(f"Valeur réelle {criteria_labels[criterion]}")
        plt.ylabel(f"Score normalisé {criteria_labels[criterion]}")
        plt.title(f"Évolution du score pour {criteria_labels[criterion]}")
        plt.legend()
        plt.grid()

    plt.tight_layout()
    plt.show()


def fonction_affine_par_morceaux(X, Y, x):
    """
    Calcule la valeur d'une fonction affine par morceaux pour un array d'abscisses x.

    Paramètres :
    X (list): Liste des abscisses des points de cassure, triée par ordre croissant.
    Y (list): Liste des ordonnées correspondantes aux points de cassure.
    x (numpy.ndarray): Array d'abscisses pour lesquelles on veut calculer la valeur de la fonction.

    Retourne :
    numpy.ndarray: Les valeurs de la fonction pour chaque point de x.
    """
    if len(X) != len(Y):
        raise ValueError("Les listes X et Y doivent avoir la même longueur.")

    # Convertir x en numpy array si ce n'est pas déjà fait
    x = np.asarray(x)

    # Initialiser un array pour stocker les résultats
    valeurs = np.zeros_like(x, dtype=float)

    # Pour chaque point dans x, calculer la valeur correspondante
    for i in range(len(x)):
        xi = x[i]
        if xi <= X[0]:
            valeurs[i] = Y[0]
        elif xi >= X[-1]:
            valeurs[i] = Y[-1]
        else:
            for j in range(1, len(X)):
                if X[j - 1] <= xi <= X[j]:
                    # Calculer les coefficients de la droite affine
                    a = (Y[j] - Y[j - 1]) / (X[j] - X[j - 1])
                    b = Y[j - 1] - a * X[j - 1]
                    # Calculer la valeur de la fonction au point xi
                    valeurs[i] = a * xi + b
                    break

    return valeurs


if __name__ == "__main__":
    preferences_DF = pd.read_excel("data\Preferences.xlsx").drop_duplicates()
    preferences = np.array(preferences_DF)[:, 1:]
    partial_ranking = []
    for i in range(len(preferences) - 1):
        partial_ranking.append([len(preferences) - (i + 1), len(preferences) - (i + 2)])

    minmax = {
        i: [np.min(preferences[:, i]), np.max(preferences[:, i])]
        for i in range(preferences.shape[1])
    }

    model, sik_matrix, turning_point_criteria = rank(
        list_alternatives=preferences,
        partial_ranking=partial_ranking,
        L=8,
        minmax=minmax,
        eps=1e-4,
    )

    model.optimize()
    score_matrix_result = sik_matrix.X
    stop_points = turning_point_criteria
    plot_score_curves(stop_points.T, score_matrix_result.T, preferences)
    values = np.zeros_like(preferences[:, 0], dtype=float)
    for i in range(preferences.shape[1]):
        values += fonction_affine_par_morceaux(
            turning_point_criteria[i, :], score_matrix_result[i, :], preferences[:, i]
        )

    preferences_DF["Score"] = values
    preferences_DF.to_excel("data/result_score.xlsx", index=False)
