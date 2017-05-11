import sys
import os
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from keras.utils.generic_utils import get_custom_objects

from stochnet.classes.NeuralNetworks import StochNeuralNetwork
from stochnet.utils.histograms import histogram_distance, get_histogram


def create_dir_if_it_does_not_exist(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def sample_from_distribution(NN, NN_prediction, nb_samples, sess=None):
    if sess is None:
        sess = tf.Session()
    samples = NN.TopLayer_obj.sample(NN_prediction, nb_samples, sess)
    return samples


def load_NN(model_folder):
    stoch_filepath = os.path.join(model_folder, 'StochNet_object.h5')
    NN = StochNeuralNetwork.load(stoch_filepath)

    keras_filepath = os.path.join(model_folder, 'keras_model.h5')
    get_custom_objects().update({"exp": lambda x: tf.exp(x),
                                 "loss_function": NN.TopLayer_obj.loss_function})
    NN.load_model(keras_filepath)
    return NN


def compute_histogram_distance(dataset_folder, NN, sess, model_id):
    settings_filepath = os.path.join(dataset_folder, 'histogram_settings.npy')
    with open(settings_filepath, 'rb') as f:
        settings = np.load(f)

    SSA_traj_filepath = os.path.join(dataset_folder, 'histogram_dataset.npy')
    with open(SSA_traj_filepath, 'rb') as f:
        SSA_traj = np.load(f)
    nb_settings = settings.shape[0]
    nb_traj = SSA_traj.shape[1]

    settings_rescaled = rescale(settings, NN.scaler)
    S_histogram_distance = np.zeros(nb_settings)
    plot_folder = os.path.join(dataset_folder, 'histogram/model_' + str(model_id))
    create_dir_if_it_does_not_exist(plot_folder)

    for i in range(nb_settings):
        S_NN_hist = get_S_hist_from_NN(settings_rescaled[i], nb_traj, sess)

        S_samples_SSA = SSA_traj[i, :, -1, 1]
        S_SSA_hist = get_histogram(S_samples_SSA, -0.5, 200.5, 201)

        make_and_save_plot(i, S_NN_hist, S_SSA_hist, plot_folder)
        S_histogram_distance[i] = histogram_distance(S_NN_hist, S_SSA_hist, 1)

    S_mean_histogram_distance = np.mean(S_histogram_distance)
    log_filepath = os.path.join(plot_folder, 'log.txt')
    with open(log_filepath, 'w') as f:
        f.write('The mean histogram distance, computed on {0} settings, is:'.format(nb_settings))
        f.write('{0}'.format(str(S_mean_histogram_distance)))

    return S_mean_histogram_distance


def get_S_hist_from_NN(setting, nb_traj, sess):
    NN_prediction = NN.predict(setting.reshape(1, 1, -1))
    NN_samples_rescaled = sample_from_distribution(NN, NN_prediction, nb_traj, sess)
    NN_samples = scale_back(NN_samples_rescaled, NN.scaler)
    S_samples_NN = NN_samples[:, 0, 0]
    S_NN_hist = get_histogram(S_samples_NN, -0.5, 200.5, 201)
    return S_NN_hist


def make_and_save_plot(figure_index, NN_hist, SSA_hist, folder):
    fig = plt.figure(figure_index)
    plt.plot(NN_hist, label='NN')
    plt.plot(SSA_hist, label='SSA')
    plt.legend()
    plot_filepath = os.path.join(folder, str(figure_index) + '.png')
    plt.savefig(plot_filepath, bbox_inches='tight')
    plt.close()
    return


def rescale(v, scaler):
    v_shape = v.shape
    flat_v = v.reshape(-1, v_shape[-1])
    flat_v_rescaled = scaler.transform(flat_v)
    v_rescaled = flat_v_rescaled.reshape(v_shape)
    return v_rescaled


def scale_back(v, scaler):
    v_shape = v.shape
    flat_v = v.reshape(-1, v_shape[-1])
    flat_v_rescaled = scaler.inverse_transform(flat_v)
    v_rescaled = flat_v_rescaled.reshape(v_shape)
    return v_rescaled


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    sess = tf.Session()

    timestep = float(sys.argv[1])
    nb_past_timesteps = int(sys.argv[2])
    training_dataset_id = int(sys.argv[3])
    validation_dataset_id = int(sys.argv[4])
    model_id = int(sys.argv[5])
    project_folder = str(sys.argv[6])

    model_folder = os.path.join(project_folder, 'models/' +
                                                str(timestep) + '/' +
                                                str(model_id))
    NN = load_NN(model_folder)

    train_folder = os.path.join(project_folder, 'dataset/data/' +
                                                str(timestep) + '/' +
                                                str(training_dataset_id))
    mean_train_hist_dist = compute_histogram_distance(train_folder, NN, sess,
                                                      model_id)

    val_folder = os.path.join(project_folder, 'dataset/data/' +
                                              str(timestep) + '/' +
                                              str(validation_dataset_id))
    mean_val_hist_dist = compute_histogram_distance(val_folder, NN, sess, model_id)
