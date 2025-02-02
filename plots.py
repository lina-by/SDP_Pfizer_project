import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import pandas as pd
from sklearn.manifold import MDS
from matplotlib.patches import Wedge

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
    This function generates the color dictionary mapping a given group of bricks covered by multiple SRs to a proportion of colors.
    Input:
            attribution:    Attribution dictionary. The keys are the SR IDs, and the values contain the list of assigned bricks.
    """
    color_list = list(mcolors.TABLEAU_COLORS.values())
    color_attribution = {}
    
    for sr_id, data in attribution.items():
        for city, proportion in data["Assigned bricks"]:
            if city not in color_attribution:
                color_attribution[city] = []
            color_attribution[city].append((color_list[sr_id % len(color_list)], proportion))
    
    return color_attribution, color_list

def plot_pie_chart(ax, x, y, colors):
    """Plots a small pie chart at a given location."""
    total = sum(proportion for _, proportion in colors)
    start_angle = 0
    
    for color, proportion in colors:
        angle = 360 * (proportion / total)
        wedge = Wedge((x, y), 0.4, start_angle, start_angle + angle, color=color)
        ax.add_patch(wedge)
        start_angle += angle


def plot_cities_attribution(attribution: dict) -> None:
    """
    This function plots the map of the SRs attribution with pie charts for cities covered by multiple SRs.
    Inputs:
            attribution:    Attribution dictionary. The keys are the SR IDs, and the values contain the list of assigned bricks.
    """
    distances = np.array(
        pd.read_excel("data/distances_villes.xlsx", header=0, index_col=0)
    )
    positions = get_positions_city(distances)
    color_attribution, color_list = get_color_attribution(attribution)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for city, (x, y) in enumerate(positions):
        if city in color_attribution:
            plot_pie_chart(ax, x, y, color_attribution[city])
    
    for i, (x, y) in enumerate(positions):
        sr_id = [sr for sr, data in attribution.items() if data["Center brick"] == i]
        text_color = 'black' if sr_id == [] else color_list[sr_id[0]]
        plt.text(
            x - 0.2,
            y + 0.2,
            f"{i+1}",
            fontsize=8,
            ha="right",
            weight="bold"
            if i in [attribution[i]["Center brick"] for i in attribution]
            else None,
            color=text_color,
        )
    
    # Add legend for SR colors
    legend_patches = [mpatches.Patch(color=color_list[sr], label=f"SR {sr+1}") for sr in range(len(color_list))][:len(attribution)]
    plt.legend(handles=legend_patches, title="Sales Reps", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title("Attribution des villes aux commerciaux")
    plt.axis("equal")
    plt.show()

if __name__ == "__main__":
    brick_distance = pd.read_csv("data/brick_rp_distances.csv", delimiter=",", header=0)
    index_values = pd.read_csv("data/bricks_index_values.csv", delimiter=",", header=0)

    current_assignment = {
        0: {"Center brick": 3, "Assigned bricks": [(3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (14, 1)]},
        1: {"Center brick": 13, "Assigned bricks": [(9, 1), (10, 1), (11, 1), (12, 1), (13, 1)]},
        2: {"Center brick": 15, "Assigned bricks": [(8, 1), (15, 1), (16, 1), (17, 1)]},
        3: {"Center brick": 21, "Assigned bricks": [(0, 1), (1, 1), (2, 1), (18, 1), (19, 1), (20, 1), (21, 1)]},
    }
    plot_cities_attribution(current_assignment)