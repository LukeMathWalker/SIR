import sys
from stochnet.classes.NeuralNetworks import StochNeuralNetwork
from stochnet.classes.TopLayers import MixtureOutputLayer, MultivariateNormalDiagOutputLayer
from stochnet.utils.iterator import NumpyArrayIterator
from stochnet.utils.file_organization import ProjectFileExplorer, get_rescaled_dataset
from stochnet.utils.utils import change_scaling
from keras.layers import Input, Dense, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.constraints import maxnorm


def get_NN(nb_past_timesteps, nb_features):
    input_tensor = Input(shape=(nb_past_timesteps, nb_features))
    flatten1 = Flatten()(input_tensor)
    NN_body = Dense(500, kernel_constraint=maxnorm(3), activation='relu')(flatten1)

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

    project_explorer = ProjectFileExplorer(project_folder)
    train_explorer = project_explorer.get_DatasetFileExplorer(timestep, train_dataset_id)
    val_explorer = project_explorer.get_DatasetFileExplorer(timestep, train_dataset_id)
    rescaled_x_train, rescaled_y_train, scaler_train = get_rescaled_dataset(train_explorer)
    rescaled_x_val, rescaled_y_val, scaler_val = get_rescaled_dataset(val_explorer)

    nb_features = rescaled_x_train.shape[-1]
    rescaled_x_val = change_scaling(rescaled_x_val, scaler_val, scaler_train)
    rescaled_y_val = change_scaling(rescaled_y_val, scaler_val, scaler_train)

    batch_size = 64
    training_generator = NumpyArrayIterator(rescaled_x_train, rescaled_y_train,
                                            batch_size=batch_size,
                                            shuffle=True)
    validation_generator = NumpyArrayIterator(rescaled_x_val, rescaled_y_val,
                                              batch_size=batch_size,
                                              shuffle=True)

    NN = get_NN(nb_past_timesteps, nb_features)
    NN.memorize_scaler(scaler_train)

    model_explorer = project_explorer.get_ModelFileExplorer(timestep, model_id)

    callbacks = []
    callbacks.append(EarlyStopping(monitor='val_loss',
                                   patience=3,
                                   verbose=1,
                                   mode='min'))

    checkpoint_filepath = model_explorer.weights_fp
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
    NN.save_model(model_explorer.keras_fp)
    NN.save(model_explorer.StochNet_fp)
