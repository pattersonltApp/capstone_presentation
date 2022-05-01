import os
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report,confusion_matrix
from keras.utils import plot_model


def load_file(filepath):
    df = pd.read_csv(filepath, header=None)
    return df.values


def load_group(directory):
    """
    load_group(directory)
    Loads all files in a directory and appends them to a list
        which gets converted into an np array and returned.
    """
    data = []
    labels = []
    for filename in os.listdir(directory):
        if os.path.isfile(directory + '/' + filename):
            values = load_file(directory + '/' + filename)
            data.append(values)
            labels.append(filename[0])
    data = np.array(data)
    labels = np.array([ord(label) - 97 for label in labels])
    return data, labels


def main():
    model = load_model('models/model_4-30.h5')
    model.summary()

    # Load test data.
    test_directory = 'BA_data/cleaned_data/test'
    test, test_labels = load_group(test_directory)

    test2, test_labels2 = load_group('MJ_data/cleaned_data/test')
    test = np.concatenate((test, test2))
    test_labels = np.concatenate((test_labels, test_labels2))

    score = model.evaluate(test, test_labels, batch_size=25, verbose=1)
    print('%s: %.2f%%' % (model.metrics_names[1], score[1]*100))

    print('TESTING REPORT:')
    predictions = model.predict_classes(test)
    predictions = predictions.reshape(1, -1)[0]
    print(classification_report(test_labels, predictions, target_names=['A','B','C','D','E','F','G','H','I','J','K','L',
                                                                        'M','N','O','P','Q','R','S','T','U','V','W','X',
                                                                        'Y','Z']))


if __name__ == '__main__':
    main()
