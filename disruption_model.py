import gurobipy
import csv

current_assignment = {
    1: [],
    2: [],
    3: [],
    4: []
}


def read_brick_values(path: str = 'data/bricks_index_values.csv'):
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        index_values = {}
        for line in reader:
            if line[0] != 'brick':
                index_values[int(line[0])] = float(line[1])
        return index_values


if __name__ == "__main__":
    print(read_brick_values())
