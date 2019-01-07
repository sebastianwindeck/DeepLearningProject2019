# Comment AS: die Funktionen sind sinnvoll; tendenziell sollten wir die aber nur für die "end-Auswertung" brauchen. Fürs
# Bewerten der Noise-Kandidaten können wir evtl. direkt eine evaluate-Funktion des Modells (aus Keras) nutzen.
# TODO: Malte check.
def final_score(y_pred, y_true):
    # TODO: [Sebastian] Create function to use several scoring functions, f1, accuracy, precision

    pass

def pitch_confussion(y_pred, y_true):
    # TODO: [Sebastian] Create a confusion matrix to get the confusion of pitches over a song
    pass