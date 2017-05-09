import os
import luigi
import luigi.contrib.external_program


class GenerateDataset(luigi.contrib.external_program.ExternalPythonProgramTask):

    project_folder = luigi.Parameter(default='/home/lucap/Documenti/Tesi Magistrale/SIR')
    dataset_id = luigi.IntParameter(default=1)
    nb_of_settings = luigi.IntParameter(default=2)
    nb_of_trajectories = luigi.IntParameter(default=100)
    timestep = luigi.FloatParameter(default=2**(-1))
    endtime = luigi.IntParameter(default=5)

    virtualenv = '/home/lucap/anaconda3/envs/py2'

    def program_args(self):
        dataset_folder = os.path.join(self.project_folder, 'dataset')
        program_address = os.path.join(dataset_folder,
                                       'multistep_simulation_w_gillespy.py')
        data_folder = os.path.join(dataset_folder, 'data')
        return ['python', program_address, self.dataset_id, self.nb_of_settings,
                self.nb_of_trajectories, self.timestep, self.endtime, data_folder]

    def output(self):
        dataset_filename = str(self.dataset_id) + '.npy'
        folder_address = os.path.join(self.project_folder, 'dataset/data/' +
                                                           str(self.timestep) + '/' +
                                                           str(self.dataset_id))
        output_address = os.path.join(folder_address, dataset_filename)
        return luigi.LocalTarget(output_address)


class FormatDataset(luigi.contrib.external_program.ExternalPythonProgramTask):

    project_folder = luigi.Parameter(default='/home/lucap/Documenti/Tesi Magistrale/SIR')
    dataset_id = luigi.IntParameter(default=1)
    timestep = luigi.FloatParameter(default=2**(-1))
    nb_past_timesteps = luigi.IntParameter(default=1)

    def requires(self):
        return GenerateDataset(project_folder=self.project_folder,
                               dataset_id=self.dataset_id,
                               timestep=self.timestep)

    def program_args(self):
        program_address = os.path.join(self.project_folder,
                                       'utils/format_for_ML_w_numpy.py')
        return ['python', program_address, self.nb_past_timesteps,
                self.dataset_id, self.timestep, self.project_folder]

    def output(self):
        dataset_folder = os.path.join(self.project_folder, 'dataset/data/' +
                                                           str(self.timestep) + '/' +
                                                           str(self.dataset_id))
        x_filepath = os.path.join(dataset_folder, str(self.dataset_id) + '_x_rescaled.npy')
        y_filepath = os.path.join(dataset_folder, str(self.dataset_id) + '_y_rescaled.npy')
        scaler_filepath = os.path.join(dataset_folder, str(self.dataset_id) + '_scaler.h5')
        return [luigi.LocalTarget(x_filepath),
                luigi.LocalTarget(y_filepath),
                luigi.LocalTarget(scaler_filepath)]


class TrainNN(luigi.contrib.external_program.ExternalPythonProgramTask):

    project_folder = luigi.Parameter(default='/home/lucap/Documenti/Tesi Magistrale/SIR')
    training_dataset_id = luigi.IntParameter(default=1)
    validation_dataset_id = luigi.IntParameter(default=2)
    timestep = luigi.FloatParameter(default=2**(-1))
    nb_past_timesteps = luigi.IntParameter(default=1)
    model_id = luigi.IntParameter(default=1)

    def requires(self):
        return [FormatDataset(project_folder=self.project_folder,
                              dataset_id=self.training_dataset_id,
                              timestep=self.timestep,
                              nb_past_timesteps=self.nb_past_timesteps),
                FormatDataset(project_folder=self.project_folder,
                              dataset_id=self.validation_dataset_id,
                              timestep=self.timestep,
                              nb_past_timesteps=self.nb_past_timesteps)]

    def program_args(self):
        program_address = os.path.join(self.project_folder,
                                       'model_training_w_numpy.py')
        return ['python', program_address, self.timestep,
                self.nb_past_timesteps, self.training_dataset_id,
                self.validation_dataset_id, self.project_folder,
                self.model_id]

    def output(self):
        model_folder = os.path.join(self.project_folder, 'models/' +
                                                         str(self.timestep) + '/' +
                                                         str(self.model_id))
        weights_filepath = os.path.join(model_folder, 'best_weights.h5')
        keras_filepath = os.path.join(model_folder, 'keras_model.h5')
        StochNet_filepath = os.path.join(model_folder, 'StochNet_object.h5')
        return [luigi.LocalTarget(weights_filepath),
                luigi.LocalTarget(keras_filepath),
                luigi.LocalTarget(StochNet_filepath)]


class GenerateHistogramData(luigi.contrib.external_program.ExternalPythonProgramTask):

    project_folder = luigi.Parameter(default='/home/lucap/Documenti/Tesi Magistrale/SIR')
    dataset_id = luigi.IntParameter(default=1)
    timestep = luigi.FloatParameter(default=2**(-1))
    nb_past_timesteps = luigi.IntParameter(default=1)
    nb_histogram_settings = luigi.IntParameter(default=15)
    nb_trajectories = luigi.IntParameter(default=500)

    virtualenv = '/home/lucap/anaconda3/envs/py2'

    def requires(self):
        return FormatDataset(project_folder=self.project_folder,
                             dataset_id=self.dataset_id,
                             timestep=self.timestep,
                             nb_past_timesteps=self.nb_past_timesteps)

    def program_args(self):
        program_address = os.path.join(self.project_folder,
                                       'utils/generator_for_histogram_w_gillespy.py')
        return ['python', program_address, self.timestep,
                self.nb_past_timesteps, self.dataset_id,
                self.nb_histogram_settings, self.nb_trajectories,
                self.project_folder]

    def output(self):
        dataset_folder = os.path.join(self.project_folder, 'dataset/data/' +
                                                           str(self.timestep) + '/' +
                                                           str(self.dataset_id))
        histogram_settings_filepath = os.path.join(dataset_folder, 'histogram_settings.npy')
        histogram_dataset_filepath = os.path.join(dataset_folder, 'histogram_dataset.npy')
        return [luigi.LocalTarget(histogram_settings_filepath),
                luigi.LocalTarget(histogram_dataset_filepath)]


class HistogramDistance(luigi.contrib.external_program.ExternalPythonProgramTask):

    project_folder = luigi.Parameter(default='/home/lucap/Documenti/Tesi Magistrale/SIR')
    training_dataset_id = luigi.IntParameter(default=1)
    validation_dataset_id = luigi.IntParameter(default=2)
    timestep = luigi.FloatParameter(default=2**(-1))
    nb_past_timesteps = luigi.IntParameter(default=1)
    model_id = luigi.IntParameter(default=1)
    nb_histogram_settings = luigi.IntParameter(default=15)
    nb_trajectories = luigi.IntParameter(default=500)

    def requires(self):
        return [TrainNN(project_folder=self.project_folder,
                        training_dataset_id=self.training_dataset_id,
                        validation_dataset_id=self.validation_dataset_id,
                        timestep=self.timestep,
                        nb_past_timesteps=self.nb_past_timesteps,
                        model_id=self.model_id),
                GenerateHistogramData(project_folder=self.project_folder,
                                      dataset_id=self.training_dataset_id,
                                      timestep=self.timestep,
                                      nb_past_timesteps=self.nb_past_timesteps,
                                      nb_histogram_settings=self.nb_histogram_settings),
                GenerateHistogramData(project_folder=self.project_folder,
                                      dataset_id=self.validation_dataset_id,
                                      timestep=self.timestep,
                                      nb_past_timesteps=self.nb_past_timesteps,
                                      nb_histogram_settings=self.nb_histogram_settings)]

    def program_args(self):
        program_address = os.path.join(self.project_folder,
                                       'metrics/histogram_w_gillespy.py')
        return ['python', program_address, self.timestep,
                self.nb_past_timesteps, self.training_dataset_id,
                self.validation_dataset_id, self.model_id,
                self.project_folder]


if __name__ == '__main__':
    luigi.run(main_task_cls=TrainNN)
