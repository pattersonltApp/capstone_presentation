import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import PReLU
from keras.layers.convolutional import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
import wandb
from wandb.keras import WandbCallback

output_path = 'models/'


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
    # Load training data.
    train_directory = 'BA_data/cleaned_data/train'
    train, train_labels = load_group(train_directory)

    train2, train_labels2 = load_group('MJ_data/cleaned_data/train')
    train = np.concatenate((train, train2))
    train_labels = np.concatenate((train_labels, train_labels2))

    train3, train_labels3 = load_group('LP_data/cleaned_data/train')
    train = np.concatenate((train, train3))
    train_labels = np.concatenate((train_labels, train_labels3))

    # Load test data.
    test_directory = 'BA_data/cleaned_data/test'
    test, test_labels = load_group(test_directory)

    test2, test_labels2 = load_group('MJ_data/cleaned_data/test')
    test = np.concatenate((test, test2))
    test_labels = np.concatenate((test_labels, test_labels2))

    test3, test_labels3 = load_group('LP_data/cleaned_data/test')
    test = np.concatenate((test, test3))
    test_labels = np.concatenate((test_labels, test_labels3))


    defaults = dict(
        dropout_1=0.2,
        dropout_2=0.2,
        dropout_3=0.2,
        hidden_layer_size=26,
        layer_1_size=32,
        layer_2_size=64,
        epochs=2500,
        batch_size=51,
        kernel_size_1=2,
        kernel_size_2=4
    )
    wandb.init(project="my-test-project", entity="pattersonlt", config=defaults)
    config = wandb.config

    optimizer = 'Adamax'

    model = Sequential()
    model.add(Conv1D(filters=config.layer_1_size, kernel_size=config.kernel_size_1, activation='relu', input_shape=(4, 1000)))
    model.add(BatchNormalization())
    act = PReLU()
    model.add(act)
    model.add(Dropout(config.dropout_1))

    model.add(MaxPooling1D(padding='same'))

    model.add(Conv1D(filters=config.layer_2_size, kernel_size=config.kernel_size_2, activation='relu', padding='same', input_shape=(4, 1000)))
    model.add(BatchNormalization())
    act = PReLU()
    model.add(act)
    model.add(Dropout(config.dropout_2))

    model.add(MaxPooling1D(padding='same'))

    model.add(Dense(config.hidden_layer_size))
    act = PReLU()
    model.add(act)
    model.add(BatchNormalization())
    model.add(Dropout(config.dropout_3))

    model.add(Flatten())

    model.add(Dense(26, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    log_dir = "logs/fit/4-21/"

    # setup model checkpointing
    model_checkpoint = ModelCheckpoint(
        filepath='C:/Users/Luke/Downloads/capstone_emg-andrew/models/model_4-21.h5',  # always overwrite the existing model
        save_weights_only=False,
        save_best_only=True, monitor='val_accuracy', verbose=1)  # only save models that improve the 'monitored' value
    callbacks = [model_checkpoint, WandbCallback()]

    model.fit(train, train_labels, validation_data=(test, test_labels), epochs=config.epochs, batch_size=config.batch_size, verbose=1, callbacks=callbacks)
    _, accuracy = model.evaluate(test, test_labels, batch_size=25, verbose=1)
    print(accuracy)


if __name__ == '__main__':
    main()
