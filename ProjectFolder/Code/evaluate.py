import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix


# Comment AS: die Funktionen sind sinnvoll; tendenziell sollten wir die aber nur für die "end-Auswertung" brauchen. Fürs
# Bewerten der Noise-Kandidaten können wir evtl. direkt eine evaluate-Funktion des Modells (aus Keras) nutzen.
# TODO: Malte check.


def final_score(y_pred, y_true):
    # TODO: [Sebastian] Create function to use several scoring functions, f1, accuracy, precision

    pass


def pitch_confusion(y_pred, y_true, type='heat'):
    # TODO: [Sebastian] Create a confusion matrix to get the confusion of pitches over a song
    _true = []
    _pred = []
    sample_weight = []
    # compare pred with true
    # number of true notes per time steps
    true_count = np.count_nonzero(y_true, axis=1)
    # number of pred notes per time steps
    pred_count = np.count_nonzero(y_pred, axis=1)
    # ratio of true count to pred count
    pred_weight = np.divide(true_count, pred_count)
    pred_weight[pred_weight == np.inf] = 1
    pred_weight[pred_weight == 0] = 1
    print(pred_weight)
    if not len(true_count) == len(pred_count):
        print("Warning evaluation will collapse due to different lenth of predicted and true label.")

    for i in range(y_pred.shape[0]):

        # Identify the notes on the piano roll
        ix_p = np.isin(y_pred[i], 1)
        print(ix_p)
        print(np.where(ix_p))
        ind_p = np.where(ix_p)[0]
        ix_t = np.isin(y_true[i], 1)
        ind_t = np.where(ix_t)[0]

        # find right classified pitches
        classified = list(set(ind_t).intersection(set(ind_p)))
        # find missed pitches
        missed = list(set(ind_t) - set(ind_p))
        # find misclassified pitches
        misclassified = list(set(ind_p) - set(ind_t))

        # Add classified pitches
        _true.extend(classified)
        _pred.extend(classified)
        weight = 0

        j = 0
        while j < len(classified):
            sample_weight.append(np.minimum(pred_weight[i], 1))
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
        # Case 4: to few predictions
        elif len(missed) > 0 and len(misclassified) > 0:
            perm = list(itertools.product(missed, misclassified))
            weight = len(missed) / len(misclassified)

        for item in perm:
            _true.append(item[0])
            _pred.append(item[1])
            sample_weight.append(weight)

    data = confusion_matrix(y_true=_true, y_pred=_pred, sample_weight=sample_weight)

    # visualize data
    if type == 'heat':
        sns.heatmap(data=data)
    elif type == 'cluster':
        sns.clustermap(data=data)
    elif type == 'joint':
        sns.jointplot(x=_true, y=_pred).plot_joint(sns.kdeplot, zorder=0, n_levels=6).set_axis_labels("True", "Pred")
    elif type == 'scatter':
        sns.scatterplot(x=_true, y=_pred, size=sample_weight)
    else:
        print("Warning the selected visualization type does not exists. "
              "Please select either 'heat' or 'cluster' for type.")

    print("Confusion Matrix done.")
    plt.show()
