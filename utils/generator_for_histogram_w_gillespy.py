import gillespy
import os
import sys
import dill
import math
import numpy as np
from tqdm import tqdm


class SIR(gillespy.Model):
    """
    This is a simple example for mass-action degradation of species S.
    """

    def __init__(self, endtime, timestep):

        # Initialize the model.
        gillespy.Model.__init__(self, name="SIR")

        # Parameters
        beta = gillespy.Parameter(name='beta', expression='3.')
        gamma = gillespy.Parameter(name='gamma', expression='1.')
        self.add_parameter([beta, gamma])

        # Species
        S = gillespy.Species(name='S', initial_value=100)
        I = gillespy.Species(name='I', initial_value=100)
        R = gillespy.Species(name='R', initial_value=100)
        self.add_species([S, I, R])

        # Reactions
        infection = gillespy.Reaction(name='infection',
                                      reactants={S: 1, I: 1},
                                      products={I: 2},
                                      propensity_function='beta*S*I/(S+I+R)')
        recover = gillespy.Reaction(name='recover',
                                    reactants={I: 1},
                                    products={R: 1},
                                    rate=gamma)
        self.add_reaction([infection, recover])
        nb_of_steps = int(math.ceil((endtime / timestep))) + 1
        self.timespan(np.linspace(0, endtime, nb_of_steps))

    def set_species_initial_value(self, species_initial_value):
        self.listOfSpecies['S'].initial_value = species_initial_value[0]
        self.listOfSpecies['I'].initial_value = species_initial_value[1]
        self.listOfSpecies['R'].initial_value = species_initial_value[2]
        return


def get_histogram_settings(nb_histogram_settings, dataset_folder, dataset_id):
    x_filepath = os.path.join(dataset_folder, str(dataset_id) + '_x_rescaled.npy')
    with open(x_filepath, 'rb') as f:
        x_data = np.load(f)

    scaler_filepath = os.path.join(dataset_folder, str(dataset_id) + '_scaler.h5')
    with open(scaler_filepath, 'rb') as f:
        scaler = dill.load(f)

    nb_samples = x_data.shape[0]
    settings_index = list(np.random.randint(low=0, high=nb_samples - 1,
                                            size=nb_histogram_settings))
    settings_rescaled = x_data[settings_index, 0, :]
    settings_shape = settings_rescaled.shape
    flat_settings_rescaled = settings_rescaled.reshape(-1, settings_shape[-1])
    flat_settings = scaler.inverse_trasform(flat_settings_rescaled)
    settings = flat_settings.reshape(settings_shape)
    return settings


def simulate(settings, nb_settings, nb_trajectories, dataset_folder, prefix='histogram_partial_'):
    for j in tqdm(range(nb_settings)):
        initial_values = settings[j]
        single_simulation(initial_values, nb_trajectories, dataset_folder, prefix, j)
    return


def single_simulation(initial_values, nb_trajectories, dataset_folder, prefix, id_number):
    model.set_species_initial_value(initial_values)
    trajectories = model.run(number_of_trajectories=nb_trajectories, show_labels=False)
    dataset = np.array(trajectories)
    save_simulation_data(dataset, dataset_folder, prefix, id_number)
    return


def save_simulation_data(dataset, dataset_folder, prefix, id_number):
    partial_dataset_filename = str(prefix) + str(id_number) + '.npy'
    partial_dataset_filepath = os.path.join(dataset_folder, partial_dataset_filename)
    with open(partial_dataset_filepath, 'wb') as f:
        np.save(f, dataset)
    return


def concatenate_simulations(nb_settings, dataset_folder, prefix='histogram_partial_'):
    for i in tqdm(range(nb_settings)):
        partial_dataset_filename = str(prefix) + str(i) + '.npy'
        partial_dataset_filepath = os.path.join(dataset_folder, partial_dataset_filename)
        with open(partial_dataset_filepath, 'rb') as f:
            partial_dataset = np.load(f)
        if i == 0:
            final_dataset = partial_dataset
        else:
            final_dataset = np.concatenate((final_dataset, partial_dataset), axis=0)
        os.remove(partial_dataset_filepath)
    return final_dataset


if __name__ == '__main__':

    timestep = float(sys.argv[1])
    nb_past_timesteps = int(sys.argv[2])
    dataset_id = int(sys.argv[3])
    nb_histogram_settings = int(sys.argv[4])
    nb_trajectories = int(sys.argv[5])
    project_folder = str(sys.argv[6])

    dataset_folder = os.path.join(project_folder, 'dataset/data/' +
                                                  str(timestep) + '/' +
                                                  str(dataset_id))

    model = SIR(endtime=nb_past_timesteps * timestep, timestep=timestep)

    settings = get_histogram_settings(nb_histogram_settings, dataset_folder, dataset_id)
    histogram_settings_filepath = os.path.join(dataset_folder, 'histogram_settings.npy')
    with open(histogram_settings_filepath, 'wb') as f:
        np.save(f, settings)

    simulate(settings, nb_histogram_settings, nb_trajectories, dataset_folder, prefix='partial_')
    histogram_dataset = concatenate_simulations(nb_histogram_settings, dataset_folder, prefix='partial_')

    histogram_dataset_filepath = os.path.join(dataset_folder, 'histogram_dataset.npy')
    with open(histogram_dataset_filepath, 'wb') as f:
        np.save(f, histogram_dataset)
