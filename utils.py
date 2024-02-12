import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(scores, figure_file, title):
    plt.plot(list(range(len(scores))), scores)
    plt.title(title)
    plt.savefig(figure_file)
    plt.pause(0.05)
