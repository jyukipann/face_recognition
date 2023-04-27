from matplotlib import pyplot as plt
import numpy as np
import csv


def load_csv(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        data = list(reader)
    return data


def load_landmarks(file_path):
    ixyz = load_csv(file_path)
    ixyz = np.array(ixyz)
    return ixyz


colorlist = ["r", "g", "b", "c", "m", "y", "k"]
def plot_landmarks(file_path, ax):
    global colorlist

    ixyz = load_landmarks(file_path).astype(float)

    x, y, z = ixyz[:, 1], ixyz[:, 2], ixyz[:, 3]

    ax.scatter(x, y, z, c=colorlist.pop())


fig = plt.figure()

ax = fig.add_subplot(1, 1, 1, projection='3d')
file_path = r"tmp\0_juki\0_juki_20230427190041.csv"
plot_landmarks(file_path, ax)
file_path = r"tmp\0_juki\0_juki_20230427190043.csv"
plot_landmarks(file_path, ax)
# file_path = r"tmp\landmarks_20230427145744.csv"
# plot_landmarks(file_path, ax)
plt.show()
