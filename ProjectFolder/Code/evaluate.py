import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix


# Comment AS: die Funktionen sind sinnvoll; tendenziell sollten wir die aber nur für die "end-Auswertung" brauchen. Fürs
# Bewerten der Noise-Kandidaten können wir evtl. direkt eine evaluate-Funktion des Modells (aus Keras) nutzen.
# TODO: Malte check.


def final_score(y_pred, y_true):

    if not y_pred.shape == y_true.shape:
        print("Warning y_pred and y_true do not have the same shape please set other ")
    # TODO: [Sebastian] Create function to use several scoring functions, f1, accuracy, precision
    #                   distribution / histogram of precision etc. over frames, possibly filtered e.g. by number of true notes in that frame.

    pass


def pitch_confusion(y_pred, y_true, vtype='heat'):
    data = np.zeros((y_pred.shape[1],y_true.shape[1]))

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
        print(i)
        # Identify the notes on the piano roll
        ix_p = np.isin(y_pred[i], 1)
        ind_p = np.where(ix_p)[0]
        ix_t = np.isin(y_true[i], 1)
        ind_t = np.where(ix_t)[0]

        # find right classified pitches
        classified = np.intersect1d(ind_t, ind_p)
        # find missed pitches
        missed = np.setdiff1d(ind_t,ind_p)
        # find misclassified pitches
        misclassified = np.setdiff1d(ind_p,ind_t)

        # Add classified pitches
        weight = 0

        j = 0
        while j < len(classified):
            data[classified[j], classified[j]] += np.minimum(pred_weight[i], 1)
            j = +1

        # Case 1: perfect silence
        if len(classified) == 0 and len(misclassified) == 0 and len(missed) == 0:
            perm = []
            weight = 0
        # Case 2: to many predictions
        elif len(missed) == 0 and len(misclassified) > 0:
            perm = list(itertools.product(classified, misclassified))
            if len(classified) > 0:
                weight = len(misclassified) / len(classified)
            else:
                weight = 1
        # Case 3: to few predictions
        elif len(missed) > 0 and len(misclassified) == 0:
            perm = list(itertools.product(missed, classified))
            if len(classified) > 0:
                weight = len(missed) / len(classified)
            else:
                weight = 1
        # Case 4: both missed and misclassified
        elif len(missed) > 0 and len(misclassified) > 0:
            perm = list(itertools.product(missed, misclassified))
            weight = len(missed) / len(misclassified)

        for item in perm:

            data[item[0], item[1]] += weight

    # visualize data
    true = []
    pred = []
    matrix_weight = []
    if vtype == 'heat':
        sns.heatmap(data=data)
    elif vtype == 'cluster':
        sns.clustermap(data=data)
    elif vtype == 'joint':
        sns.jointplot(x=true, y=pred).plot_joint(sns.kdeplot, zorder=0, n_levels=6)\
            .set_axis_labels("True", "Pred")
    elif vtype == 'scatter':
        sns.scatterplot(x=true, y=pred, size=matrix_weight)
    else:
        print("Warning the selected visualization type does not exists. "
              "Please select either 'heat' or 'cluster' for type.")

    print("Confusion Matrix done.")
    plt.show()
