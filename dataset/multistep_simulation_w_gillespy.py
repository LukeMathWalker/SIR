import gillespy
import numpy as np
import os
import sys
import math
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


def create_dir_if_it_does_not_exist(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def simulate(settings, nb_of_settings, nb_trajectories, dataset_folder, prefix='partial_'):
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


def concatenate_simulations(nb_settings, dataset_folder, prefix='partial_'):
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
    dataset_id = int(sys.argv[1])
    nb_settings = int(sys.argv[2])
    nb_trajectories = int(sys.argv[3])
    timestep = float(sys.argv[4])
    endtime = float(sys.argv[5])
    data_root_folder = str(sys.argv[6])

    dataset_folder = os.path.join(data_root_folder, str(timestep))
    dataset_folder = os.path.join(dataset_folder, str(dataset_id))
    create_dir_if_it_does_not_exist(dataset_folder)

    model = SIR(endtime, timestep)

    settings = np.random.randint(low=30, high=200, size=(nb_settings, 3))
    simulate(settings, nb_settings, nb_trajectories, dataset_folder, prefix='partial_')
    dataset = concatenate_simulations(nb_settings, dataset_folder, prefix='partial_')

    dataset_filepath = os.path.join(dataset_folder, str(dataset_id) + '.npy')
    with open(dataset_filepath, 'wb') as f:
        np.save(f, dataset)
