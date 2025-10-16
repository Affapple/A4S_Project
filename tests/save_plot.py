import pandas as pd
from matplotlib import pyplot as plt

def draw_plot(file_path, x, y, title, x_label, y_label):
    OUTPUT_FOLDER = "./tests/data/measures/"
    file_path = OUTPUT_FOLDER + file_path.lower().replace(' ', '_')

    df = pd.read_csv(file_path + ".csv")
    X = df[x] if x else range(len(df))
    Y = df[y]
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks(X)
    ax.plot(X, Y, marker='o', )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)
    fig.savefig(file_path + ".png")