import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from sklearn.manifold import MDS


def rotate_symmetry(points: np.ndarray, theta: float) -> np.ndarray:
    """
    This function applies a rotation of all points by an angle theta an then applies a vertical symmetry
    Inputs:
            points:         NumPy array containing all points with (x,y) format. Shape = [number_of_points, 2]
            theta:          Angle of the rotation in radians
    """

    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )

    rotated_points = [np.dot(rotation_matrix, point) for point in points]
    return np.array([[x, -y] for (x, y) in rotated_points])


def get_positions_city(distances: np.ndarray) -> np.ndarray:
    """
    This function generates the location of the cities based on the distances matrix between each pair of cities
    Input:
            distances:      Distance matrix between the cities
    """

    mds = MDS(
        n_components=2, dissimilarity="precomputed", random_state=42, max_iter=1000
    )
    positions = mds.fit_transform(distances)
    return rotate_symmetry(positions, -2 * np.pi / 3)


def get_color_attribution(attribution: dict) -> dict:
    """
    This function generates the color dictionnary mapping a given group of bricks covered by the same SR to a same color.
    Input:
            attribution:    Attribution dictionnary. The keys are the location of the center brick and the values are the lists of the covered cities
    """

    color_list = list(mcolors.TABLEAU_COLORS)

    color_attribution = {}
    for color_id, center in enumerate(attribution):
        for city in attribution[center]:
            color_attribution[city] = color_list[color_id]

    return color_attribution


def plot_cities_attribution(attribution: dict) -> None:
    """
    This function plots the map of the SRs attribution.
    Inputs:
            attribution:    Attribution dictionnary. The keys are the location of the center brick and the values are the lists of the covered cities
    """
    distances = np.array(
        pd.read_excel("data/distances_villes.xlsx", header=0, index_col=0)
    )
    positions = get_positions_city(distances)
    color_attribution = get_color_attribution(attribution)

    plt.figure(figsize=(8, 6))

    plt.scatter(
        positions[:, 0],
        positions[:, 1],
        color=[
            color_attribution[city] for city in range(1, len(color_attribution) + 1)
        ],
        s=50,
    )

    for i, (x, y) in enumerate(positions):
        plt.text(
            x - 0.2,
            y + 0.2,
            f"{i+1}",
            fontsize=8,
            ha="right",
            weight="bold" if i + 1 in attribution else None,
        )

    plt.title("Attribution des villes aux commerciaux")
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    brick_distance = pd.read_csv("data/brick_rp_distances.csv", delimiter=",", header=0)
    index_values = pd.read_csv("data/bricks_index_values.csv", delimiter=",", header=0)

    attribution_initiale = {
        4: [4, 5, 6, 7, 8, 15],
        14: [10, 11, 12, 13, 14],
        16: [9, 16, 17, 18],
        22: [1, 2, 3, 19, 20, 21, 22],
    }
    plot_cities_attribution(attribution_initiale)
