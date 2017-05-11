import sys
import os
import dill
import numpy as np
from tqdm import tqdm
from stochnet.classes.NeuralNetworks import StochNeuralNetwork
from stochnet.classes.TopLayers import MultivariateNormalCholeskyOutputLayer, MultivariateLogNormalOutputLayer, MixtureOutputLayer, MultivariateNormalDiagOutputLayer
from stochnet.utils.iterator import NumpyArrayIterator
from keras.layers import Input, LSTM, Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.constraints import maxnorm
from keras.regularizers import l2
import tensorflow as tf


def create_dir_if_it_does_not_exist(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def get_dataset(data_folder, dataset_id):
    dataset_folder = os.path.join(data_folder, str(dataset_id))
    x_filepath = os.path.join(dataset_folder, 'x_rescaled.npy')
    y_filepath = os.path.join(dataset_folder, 'y_rescaled.npy')
    scaler_filepath = os.path.join(dataset_folder, 'scaler.h5')
    with open(x_filepath, 'rb') as f:
        x = np.load(f)
    with open(y_filepath, 'rb') as f:
        y = np.load(f)
    with open(scaler_filepath, 'rb') as f:
        scaler = dill.load(f)
    return x, y, scaler


def change_scaling(data, old_scaler, new_scaler):
    data_shape = data.shape
    flat_data = data.reshape((-1, data_shape[-1]))
    flat_data = scaler_train.transform(scaler_val.inverse_transform(flat_data))
    data = flat_data.reshape(data_shape)
    return data


def get_NN(nb_past_timesteps, nb_features):
    input_tensor = Input(shape=(nb_past_timesteps, nb_features))
    flatten1 = Flatten()(input_tensor)
    NN_body = Dense(128, kernel_constraint=maxnorm(3), activation='relu')(flatten1)

    number_of_components = 2
    components = []
    for j in range(number_of_components):
        components.append(MultivariateNormalDiagOutputLayer(nb_features))

    TopModel_obj = MixtureOutputLayer(components)
    NN = StochNeuralNetwork(input_tensor, NN_body, TopModel_obj)
    return NN


if __name__ == '__main__':

    timestep = float(sys.argv[1])
    nb_past_timesteps = int(sys.argv[2])
    train_dataset_id = int(sys.argv[3])
    val_dataset_id = int(sys.argv[4])
    project_folder = str(sys.argv[5])
    model_id = int(sys.argv[6])

    data_folder = os.path.join(project_folder, 'dataset/data/' + str(timestep))
    x_train, y_train, scaler_train = get_dataset(data_folder, train_dataset_id)
    x_val, y_val, scaler_val = get_dataset(data_folder, val_dataset_id)

    nb_features = x_train.shape[-1]
    x_val = change_scaling(x_val, scaler_val, scaler_train)
    y_val = change_scaling(y_val, scaler_val, scaler_train)

    batch_size = 64
    training_generator = NumpyArrayIterator(x_train, y_train,
                                            batch_size=batch_size,
                                            shuffle=True)
    validation_generator = NumpyArrayIterator(x_val, y_val,
                                              batch_size=batch_size,
                                              shuffle=True)

    NN = get_NN(nb_past_timesteps, nb_features)
    NN.memorize_scaler(scaler_train)

    model_directory = os.path.join(project_folder, 'models/' +
                                                   str(timestep) + '/' +
                                                   str(model_id))
    create_dir_if_it_does_not_exist(model_directory)

    callbacks = []
    callbacks.append(EarlyStopping(monitor='val_loss',
                                   patience=3,
                                   verbose=1,
                                   mode='min'))

    checkpoint_filepath = os.path.join(model_directory, 'best_weights.h5')
    callbacks.append(ModelCheckpoint(checkpoint_filepath, monitor='val_loss',
                                     verbose=1, save_best_only=True,
                                     save_weights_only=True, mode='min'))
    result = NN.fit_generator(training_generator=training_generator,
                              samples_per_epoch=3 * 10**4, epochs=5, verbose=1,
                              callbacks=callbacks,
                              validation_generator=validation_generator,
                              nb_val_samples=10**3)
    lowest_val_loss = min(result.history['val_loss'])
    print(lowest_val_loss)

    NN.load_weights(checkpoint_filepath)
    model_filepath = os.path.join(model_directory, 'keras_model.h5')
    NN.save_model(model_filepath)

    filepath = os.path.join(model_directory, 'StochNet_object.h5')
    NN.save(filepath)
