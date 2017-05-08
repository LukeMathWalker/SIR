import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt

from keras.layers import Input, LSTM, Dense, Dropout, Flatten
from keras.constraints import maxnorm
from keras.utils.generic_utils import get_custom_objects
import keras.activations

from stochnet.classes.TimeSeriesDataset import TimeSeriesDataset
from stochnet.classes.NeuralNetworks import StochNeuralNetwork
from stochnet.classes.TopLayers import MultivariateNormalCholeskyOutputLayer, MixtureOutputLayer
from stochnet.utils.histograms import histogram_distance, get_histogram

import os
import stochpy
import pandas as pd
import shutil
import tensorflow as tf


def sample_from_distribution(NN, NN_prediction, nb_samples, sess=None):
    if sess is None:
        sess = tf.Session()
    samples = NN.TopLayer_obj.sample(NN_prediction, nb_samples, sess)
    return samples


np.set_printoptions(suppress=True)
sess = tf.Session()

nb_of_trajectories_for_hist = 10**4
nb_features = 3
time_step_size = 5. / 11.
with open('hist_settings.npy', 'rb') as f:
    initial_sequences = np.load(f)

nb_of_initial_configurations = initial_sequences.shape[0]

stoch_filepath = '/home/lucap/Documenti/Tesi Magistrale/StochNet/stochnet/models/SIR_timestep_2-1/model_02/SIR_-13.5906531482.h5'
NN = StochNeuralNetwork.load(stoch_filepath)

model_filepath = '/home/lucap/Documenti/Tesi Magistrale/StochNet/stochnet/models/SIR_timestep_2-1/model_02/model.h5'

get_custom_objects().update({"exp": lambda x: tf.exp(x),
                             "loss_function": NN.TopLayer_obj.loss_function})

NN.load_model(model_filepath)
initial_sequences_rescaled = NN.scaler.transform(initial_sequences.reshape(-1, nb_features)).reshape(nb_of_initial_configurations, -1, nb_features)
S_histogram_distance = np.zeros(nb_of_initial_configurations)

for i in range(nb_of_initial_configurations):
    print('\n\n')
    print(initial_sequences[i])
    NN_prediction = NN.predict(initial_sequences_rescaled[i][np.newaxis, :, :])
    NN_samples_rescaled = sample_from_distribution(NN, NN_prediction, nb_of_trajectories_for_hist, sess)
    NN_samples = NN.scaler.inverse_transform(NN_samples_rescaled.reshape(-1, nb_features)).reshape(nb_of_trajectories_for_hist, -1, nb_features)
    S_samples_NN = NN_samples[:, 0, 0]
    S_NN_hist = get_histogram(S_samples_NN, -0.5, 200.5, 201)
    plt.figure(i)
    plt.plot(S_NN_hist, label='NN')

    with open('histogram_dataset_' + str(i) + '.npy', 'rb') as f:
        trajectories = np.load(f)
    S_samples_SSA = trajectories[..., 1]
    S_SSA_hist = get_histogram(S_samples_SSA, -0.5, 200.5, 201)
    plt.plot(S_SSA_hist, label='SSA')
    plt.legend()
    plt.savefig('test_' + str(i) + '.png', bbox_inches='tight')

    plt.close()
    S_histogram_distance[i] = histogram_distance(S_NN_hist, S_SSA_hist, 1)
    # print("Histogram distance:")
    # print(S_histogram_distance)
print(S_histogram_distance)
print(np.mean(S_histogram_distance))
