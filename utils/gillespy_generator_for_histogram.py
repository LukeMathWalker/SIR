import gillespy
import numpy as np
from tqdm import tqdm


class SIR(gillespy.Model):
    """
    This is a simple example for mass-action degradation of species S.
    """

    def __init__(self):

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
        self.timespan(np.linspace(0, 5, 11))

    def set_species_initial_value(self, species_initial_value):
        self.listOfSpecies['S'].initial_value = species_initial_value[0]
        self.listOfSpecies['I'].initial_value = species_initial_value[1]
        self.listOfSpecies['R'].initial_value = species_initial_value[2]
        return


if __name__ == '__main__':
    model = SIR()

    nb_settings = 50
    settings = np.random.randint(low=10, high=200, size=(nb_settings, 3))
    with open('hist_settings.npy', 'wb') as f:
        np.save(f, settings)
    num_trajectories = 10**3
    for j in tqdm(range(nb_settings)):
        species_initial_value = settings[j]
        model.set_species_initial_value(species_initial_value)
        trajectories = model.run(number_of_trajectories=num_trajectories, show_labels=False)
        dataset = np.array(trajectories)
        histogram_dataset = dataset[:, 1, :]
        dataset_filepath = 'histogram_dataset_' + str(j) + '.npy'
        with open(dataset_filepath, 'wb'):
            np.save(dataset_filepath, histogram_dataset)
