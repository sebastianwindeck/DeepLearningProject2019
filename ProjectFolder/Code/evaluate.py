import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report


def final_score(y_pred, y_true, description):
    if not y_pred.shape == y_true.shape:
        print("Warning y_pred and y_true do not have the same shape please set other ")

    # compare pred with true
    # number of true notes per time steps
    true_count = np.count_nonzero(y_true, axis=1)

    for i in np.unique(true_count):
        print("Report for {} in noise {} level".format(i, description))
        print(classification_report(y_true=y_true[np.where(true_count == i)],
                                    y_pred=y_pred[np.where(true_count == i)]))  # save in variable to work with


def pitch_confusion(y_pred, y_true, save_path, description, vtype='heat'):
    global perm
    data = np.zeros((y_pred.shape[1], y_true.shape[1]))

    # compare pred with true
    # number of true notes per time steps
    true_count = np.count_nonzero(y_true, axis=1)
    # number of pred notes per time steps
    pred_count = np.count_nonzero(y_pred, axis=1)
    # ratio of true count to pred count
    pred_weight = np.divide(true_count, pred_count)
    pred_weight[pred_weight == np.inf] = 1
    pred_weight[pred_weight == 0] = 1
    if not len(true_count) == len(pred_count):
        print("Warning evaluation will collapse due to different length of predicted and true labels.")

    for i in range(y_pred.shape[0]):
        # Identify the notes on the piano roll
        ix_p = np.isin(y_pred[i], 1)
        ind_p = np.where(ix_p)[0]
        ix_t = np.isin(y_true[i], 1)
        ind_t = np.where(ix_t)[0]

        # find right classified pitches
        classified = np.intersect1d(ind_t, ind_p)
        # find missed pitches
        missed = np.setdiff1d(ind_t, ind_p)
        # find misclassified pitches
        misclassified = np.setdiff1d(ind_p, ind_t)

        # Add classified pitches
        weight = 0

        for j in range(classified.shape[0]):
            data[classified[j], classified[j]] += np.minimum(pred_weight[i], 1)

        # Case 1: perfect silence
        if len(classified) == 0 and len(misclassified) == 0 and len(missed) == 0:
            perm = []
            weight = 0
        # Case 2: to many predictions
        elif len(missed) == 0 and len(misclassified) > 0:
            perm = np.stack(np.meshgrid(classified, misclassified), -1).reshape(-1, 2)
            if len(classified) > 0:
                weight = len(misclassified) / len(classified)
            else:
                weight = 1
        # Case 3: to few predictions
        elif len(missed) > 0 and len(misclassified) == 0:
            perm = np.stack(np.meshgrid(missed, classified), -1).reshape(-1, 2)
            if len(classified) > 0:
                weight = len(missed) / len(classified)
            else:
                weight = 1
        # Case 4: both missed and misclassified
        elif len(missed) > 0 and len(misclassified) > 0:
            perm = np.stack(np.meshgrid(missed, misclassified), -1).reshape(-1, 2)
            weight = len(missed) / len(misclassified)

        for row in perm:
            data[row[0], row[1]] += weight

    # denumerate data matrix in 2d indices combination with value in column 3

    xx, yy = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    data_v = np.vstack((xx.ravel(), yy.ravel(), data.ravel())).T
    data_v = np.delete(data_v, np.where(data_v[:, 2] == 0), axis=0)

    # visualize data
    if vtype == 'heat':
        g = sns.heatmap(data=data)
        g.set_title('Diagram shows {0} map of {1} epoch'.format(vtype, description))
        fig = g.get_figure()
    elif vtype == 'cluster':
        g = sns.clustermap(data=data)
        g.fig.suptitle('Diagram shows {0} map of {1} epoch'.format(vtype, description))
        fig = g.fig
    elif vtype == 'joint':
        g = sns.jointplot(x=data_v[:, 0], y=data_v[:, 1]).plot_joint(sns.kdeplot, zorder=0, n_levels=6)\
            .set_axis_labels("True", "Pred")
        g.fig.suptitle('Diagram shows {0} map of {1} epoch'.format(vtype, description))
        fig = g.fig
    elif vtype == 'scatter':
        g = sns.scatterplot(x=data_v[:, 0], y=data_v[:, 1], size=data_v[:, 2])
        fig = g.get_figure()
    else:
        print("Warning the selected visualization type does not exists. "
              "Please select either 'heat' or 'cluster' for type.")
    fig.savefig(save_path + str('_confusion_matrix_') + str(description) + str('_epoch.png'))
    plt.close(fig)
    print("Confusion Matrix done.")
