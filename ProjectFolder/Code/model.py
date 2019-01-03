class AMTNetwork():

    def __init__(self):
        # TODO: [Andreas] define network,
        pass

    def build(self):
        # TODO: [Andreas] build network architecture
        # 	Festlegen: Parameter Anzahl
        #   	Architektur: Tiefe
        # 	    Länge Samples
        # 	    Anzahl Samples

        pass

    def train(self, data, type):
        # TODO: [Andreas] (based on some data, "clean" or "noisy")
        pass

    def transcribe(self, data):
        # TODO: [Andreas] (apply learned network to new data and generate output)
        output = []

        return output

    def evaluate(self, y_pred, y_true):
        # TODO: [Andreas]   (compare transcription with “ground truth”)

        return

    def save(self):
        # TODO: [Sebastian] Save model to output file after learning
        pass


class Noiser():
    # TODO: [Tanos] Create noise machine for several noise types and intensitiy to combine the noise frame by frame to sample

    def __init__(self):
        pass

    def calculator(self):
        pass


def featureextraction():
    # TODO: [Andreas] (basierend auf CQT aus librosa)
    output = []

    return output
