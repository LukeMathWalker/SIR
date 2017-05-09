import sys
import os
import dill
import numpy as np
from stochnet.classes.TimeSeriesDataset import TimeSeriesDataset


if __name__ == '__main__':

    nb_past_timesteps = int(sys.argv[1])
    dataset_id = int(sys.argv[2])
    timestep = float(sys.argv[3])
    project_folder = str(sys.argv[4])

    dataset_folder = os.path.join(project_folder, 'dataset/data/' +
                                                  str(timestep) + '/' +
                                                  str(dataset_id))
    timeseries_dataset_filepath = os.path.join(dataset_folder,
                                               str(dataset_id) + '.npy')
    timeseries = TimeSeriesDataset(timeseries_dataset_filepath,
                                   data_format='numpy',
                                   with_timestamps=True,
                                   labels=None)
    timeseries.format_dataset_for_ML(keep_timestamps=False,
                                     nb_past_timesteps=nb_past_timesteps,
                                     must_be_rescaled=True,
                                     positivity=None,
                                     train_test_split=False)

    x_filepath = os.path.join(dataset_folder, str(dataset_id) + '_x_rescaled.npy')
    y_filepath = os.path.join(dataset_folder, str(dataset_id) + '_y_rescaled.npy')
    scaler_filepath = os.path.join(dataset_folder, str(dataset_id) + '_scaler.h5')

    with open(x_filepath, 'wb') as f:
        np.save(f, timeseries.X_data)

    with open(y_filepath, 'wb') as f:
        np.save(f, timeseries.y_data)

    with open(scaler_filepath, 'wb') as f:
        dill.dump(timeseries.scaler, f)
