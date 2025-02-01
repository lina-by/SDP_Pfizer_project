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
    num_criteria = list_alternatives.shape[1]

    turning_point_criteria = np.array(
        [
            np.linspace(minmax[criteria][0], minmax[criteria][1], L + 1)
            for criteria in minmax
        ]
    )

    segment_identification = {}  # This dictionnary will be used to compute piecewise affine scores.
    for [id0, id1] in partial_ranking:
        segment_identification = add_id_to_segment_identification_if_necessary(
            list_alternatives, L, minmax, segment_identification, id0
        )
        segment_identification = add_id_to_segment_identification_if_necessary(
            list_alternatives, L, minmax, segment_identification, id1
        )

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
        s_ix0 = get_score(L, num_criteria, segment_identification, sik_matrix, id0)
        s_ix1 = get_score(L, num_criteria, segment_identification, sik_matrix, id1)

        model.addConstr(
            s_ix0 + sigmaplus[id0] - sigmamoins[id0]
            >= s_ix1 + sigmaplus[id1] - sigmamoins[id1] + eps,
            name=f"ranking constraints {id0}/{id1}",
        )

    model.setObjective(sigmaplus.sum() + sigmamoins.sum(), GRB.MINIMIZE)

    return model, sik_matrix, turning_point_criteria


def get_score(L, num_criteria, segment_identification, sik_matrix, id):
    """
    This function gets the score calculation of a given alternative to use in the gurobi optimisation.

    Inputs:
        L:                          Number of pieces for the piecewize-affine function of each criteria
        num_criteria:               Number of various criteria taken into account for the scoring function
        segment_identification:     Dictionnary used to map each alternative to the relevant segment of the piecewise-affine score function
        sik_matrix:                 Gurobi optimisation variable matrix determining the altitude of each turning point for each criteria
        id:                         The id of the alternative whose score is to be determined

    Output:
        score:                      The computed score expressed with Gurobi variables.
    """

    score = 0
    for i in range(num_criteria):
        if segment_identification[id][i][0] == L:
            score += sik_matrix[i, -1]

        else:  # Compute piecewise-affine score
            score += (
                segment_identification[id][i][1]
                * (
                    sik_matrix[i, segment_identification[id][i][0] + 1]
                    - sik_matrix[i, segment_identification[id][i][0]]
                )
                + sik_matrix[i, segment_identification[id][i][0]]
            )

    return score


def add_id_to_segment_identification_if_necessary(
    list_alternatives, L, minmax, segment_identification, id
):
    """
    This function adds the identification of each creteria segment for the given alternative. This will be used to compute piecewise affine scores afterwards.

    Inputs:
        list_alternatives:          Array of size (number of alternatives, number of criteria) representing the score on each criteria and for each alternative
        L:                          Number of pieces for the piecewize-affine function of each criteria
        minmax:                     Dictionnary whose keys is the id of the criteria and the value is a pair [minimum value, maximum value] for the given criteria.
        segment_identification:     Dictionnary storing the segment identifications.
        id:                         Id of the alternative to be added if it is not already in the dictionnary

    Output:
        segment_identification:     Up to date dictionnary storing the segment identifications.
    """

    if id not in segment_identification:
        segment_identification[id] = []
        values = list_alternatives[id]
        for i, value in enumerate(values):
            min, max = minmax[i]
            k = math.floor(L * (value - min) / (max - min))
            xik = min + k * (max - min) / L
            segment_identification[id].append([k, L * (value - xik) / (max - min)])

    return segment_identification


def plot_score_curves(stop_points, score_matrix_result, alternatives):
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

        # Ajouter les alternatives sur la courbe avec interpolation
        for alt_idx, alt in enumerate(alternatives):
            index = np.searchsorted(stop_points[:, criterion], alt[criterion])
            if index == 0:
                interpolated_score = score_matrix_result[0, criterion]
            elif index >= len(stop_points):
                interpolated_score = score_matrix_result[-1, criterion]
            else:
                dx = (alt[criterion] - stop_points[index - 1, criterion]) / (
                    stop_points[index, criterion] - stop_points[index - 1, criterion]
                )
                dy = dx * (
                    score_matrix_result[index, criterion]
                    - score_matrix_result[index - 1, criterion]
                )
                interpolated_score = score_matrix_result[index - 1, criterion] + dy

            plt.scatter(alt[criterion], interpolated_score, marker="x", color="red")
            plt.text(
                alt[criterion],
                interpolated_score,
                str(alt_idx),
                fontsize=9,
                verticalalignment="bottom",
            )

        plt.xlabel(f"Valeur réelle {criteria_labels[criterion]}")
        plt.ylabel(f"Score calculé {criteria_labels[criterion]}")
        plt.title(f"Score affine par Morceaux du {criteria_labels[criterion]}")
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


def compute_scoring_based_on_preferences(
    preferences, partial_ranking, L, eps, plot_curves, all_alternatives
):
    """
    This function computes a piecewise-affine scoring function to respect the given partial ranking and computes the score on all the given alternatives.

    Inputs:
        preferences:        Array of size (number of alternatives, number of criteria) representing the score on each criteria and for each alternative of the training set
        partial ranking:    List of pairs of alternatives exhibiting a preference. For example, the pair [0,1] means alternative 0 is scored higher than alternaative 1
        L:                  Number of pieces for the piecewize-affine function of each criteria
        eps:                Epsilon value determining strictly inequality constraints (a > b     <=>    a >= b + eps)
        plot_curves:        Set to True if you want to plot the piecewize affine functions for each criteria
        all_alternatives:   Array of size (number of alternatives, number of criteria) representing the score on each criteria and for each alternative of the application set

    Output:
        scores:             Array of size (number of alternatives) giving the computed score of the application set's alternatives
    """

    minmax = {
        i: [np.min(preferences[:, i]), np.max(preferences[:, i])]
        for i in range(preferences.shape[1])
    }

    model, sik_matrix, turning_point_criteria = rank(
        list_alternatives=preferences,
        partial_ranking=partial_ranking,
        L=L,
        minmax=minmax,
        eps=eps,
    )

    model.optimize()

    if model.status == GRB.OPTIMAL:
        score_matrix_result = sik_matrix.X

        if plot_curves:
            stop_points = turning_point_criteria
            plot_score_curves(stop_points.T, score_matrix_result.T, preferences)

        scores = np.zeros_like(all_alternatives[:, 0], dtype=float)

        for i in range(all_alternatives.shape[1]):
            scores += fonction_affine_par_morceaux(
                turning_point_criteria[i, :],
                score_matrix_result[i, :],
                all_alternatives[:, i],
            )

        return scores

    else:
        raise ValueError("No optimal result found!")


if __name__ == "__main__":
    preferences_DF = pd.read_excel("data\Preferences.xlsx").drop_duplicates()
    preferences = np.array(preferences_DF)[:, 1:]

    partial_ranking = []
    for i in range(len(preferences) - 1):
        partial_ranking.append([len(preferences) - (i + 1), len(preferences) - (i + 2)])

    values = compute_scoring_based_on_preferences(
        preferences, partial_ranking, 4, 1e-3, True, preferences
    )

    preferences_DF["Score"] = values
    preferences_DF.to_excel("data/result_score.xlsx", index=False)
