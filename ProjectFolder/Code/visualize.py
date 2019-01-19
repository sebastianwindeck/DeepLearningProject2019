import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import numpy as np
import seaborn as sns
from model_functions import calculating_class_weights


def visualize_weights(y_true, save_path):
    width = 1
    weights = np.log10(calculating_class_weights(y_true=y_true, type='over_columns').T)
    fig, ax = plt.subplots()
    ax.bar(np.arange(weights.shape[0]), weights[:, 0], width=width, color='IndianRed',
           label='Pitchwise balancing weight', align='edge')
    weight = np.log10(calculating_class_weights(y_true=y_true, type='over_all').T)
    ax.bar(np.arange(weight.shape[0]), weight[:, 0], width=width, color='g', label='Global balancing weight',
           linestyle='--', alpha=0.5, align='edge')

    # Visualization option 2:
    # weight = calculating_class_weights(y_true=outputs, type='over_all').T
    # weight = np.append(weight,weight[0])
    # weight = np.log10(weight)
    # ax.plot(weight, color='IndianRed', label='Global balancing weight', linestyle='--')
    ax.set_ylabel('Logarithmic balancing weight of pitches')
    ax.set_xlabel('Pitch number')
    ax.set_title('Weight comparison from pitch wise to all')
    ax.legend()
    fig.savefig(save_path)  # save the figure to file
    plt.close(fig)


def visualize_input(inputs, save_path):
    sums = np.sum(inputs, axis=1).T
    fig, ax = plt.subplots()
    ax = sns.heatmap(sums[:, :1000])
    ax.set_ylabel('CQT Bandwidth')
    ax.set_xlabel('Frames (time line)')
    ax.set_title('Midi heatmap (aggregated over windows for 1000 samples)')
    fig.savefig(save_path)  # save the figure to file
    plt.close(fig)
