import numpy as np
import os
from functions import (standardize_data, generate_datasets, plot_accuracy, plot_loss, plot_training_validation_loss, 
                       plot_accuracy_histogram, plot_conductance_map, plot_confusion_matrix, plot_validation_accuracies)
from FOM_reservoir import output_voltage
from tensorflow import keras
from keras import models, layers, optimizers, losses, constraints
from sklearn.metrics import confusion_matrix

V_bias = 0.5 # Bias voltage
num_output_voltages = len(output_voltage[0])  # Number of reservoir output voltages per pattern

# Output directories for different figures
out_dir_1 = './Problem_1/FOM Outputs/Readout Layer/Training/'  # Directory for training-related plots
out_dir_2 = './Problem_1/FOM Outputs/Readout Layer/Weights/'  # Directory for weight-related plots
out_dir_3 = './Problem_1/FOM Outputs/Readout Layer/Testing/'  # Directory for testing-related plots

# Create directories if they do not exist
os.makedirs(out_dir_1, exist_ok=True)
os.makedirs(out_dir_2, exist_ok=True)
os.makedirs(out_dir_3, exist_ok=True)

def run_single_simulation(standardize=True, save_model=False, include_bias=True):
    """
    Run a single simulation of the readout layer with optional data standardization, model saving, and bias inclusion.

    Parameters:
    - standardize (bool): Whether to standardize the datasets.
    - save_model (bool): Whether to save the trained model.
    - include_bias (bool): Whether to include a bias voltage in the dataset.

    Returns:
    - final_loss (float): Final loss value of the model.
    - final_accuracy (float): Final accuracy of the model.
    - weights (list): List of model weights.
    - pattern_accuracies_evo (dict): Accuracy per pattern over epochs during training.
    - losses_evo (list): Training and validation loss values over epochs.
    """
    # Number of datasets for training and evaluating
    num_train_datasets = 100
    num_test_datasets = 20

    # Generate datasets
    train_datasets = generate_datasets(output_voltage, num_train_datasets)
    test_datasets = generate_datasets(output_voltage, num_test_datasets)

    # Standardize datasets if specified
    if standardize:
        train_datasets_standardized, test_datasets_standardized, mean_train, std_train = standardize_data(train_datasets, test_datasets)
    else:
        train_datasets_standardized = train_datasets
        test_datasets_standardized = test_datasets

    # Flatten datasets for input to neural network
    num_samples_train = train_datasets_standardized.shape[0] * train_datasets_standardized.shape[1]
    num_samples_test = test_datasets_standardized.shape[0] * test_datasets_standardized.shape[1]
    train_datasets_flat = train_datasets_standardized.reshape(num_samples_train, -1)
    test_datasets_flat = test_datasets_standardized.reshape(num_samples_test, -1)

    # Add bias to datasets if specified
    if include_bias:
        train_datasets_with_bias = np.concatenate([train_datasets_flat, V_bias * np.ones((num_samples_train, 1))], axis=1)
        test_datasets_with_bias = np.concatenate([test_datasets_flat, V_bias * np.ones((num_samples_test, 1))], axis=1)
    else:
        train_datasets_with_bias = train_datasets_flat
        test_datasets_with_bias = test_datasets_flat

    # Training and evaluation labels
    num_classes = 4
    y_train = np.array([i % num_classes for i in range(num_train_datasets * num_classes)])
    y_test = np.array([i % num_classes for i in range(num_test_datasets * num_classes)])
   
    # Convert labels to categorical format (one-hot encoding)
    y_train_categorical = keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test_categorical = keras.utils.to_categorical(y_test, num_classes=num_classes)

    # Define and compile neural network (readout layer)
    input_dim = train_datasets_with_bias.shape[1]
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.Dense(num_classes, activation='softmax', use_bias=False, kernel_constraint=constraints.NonNeg()))
    model.compile(optimizer=optimizers.Adam(), loss=losses.CategoricalCrossentropy(), metrics=['accuracy'])

    # Initialize tracking structures
    pattern_accuracies_evo = {i: [] for i in range(num_classes)}  # Accuracy evolution
    losses_evo = []  # Loss evolution

    # Training loop
    for epoch in range(100):
        history = model.fit(train_datasets_with_bias, y_train_categorical, epochs=1, batch_size=10, validation_data=(test_datasets_with_bias, y_test_categorical), verbose=0)
        losses_evo.append({'loss': history.history['loss'][0], 'val_loss': history.history['val_loss'][0]})
        
        val_predictions = model.predict(test_datasets_with_bias, verbose=0)
        val_predicted_classes = np.argmax(val_predictions, axis=1)
        
        for pattern in range(num_classes):
            pattern_mask = (y_test == pattern)
            pattern_accuracy = np.mean(val_predicted_classes[pattern_mask] == pattern)
            pattern_accuracies_evo[pattern].append(pattern_accuracy)

    # Final model evaluation
    final_loss, final_accuracy = model.evaluate(test_datasets_with_bias, y_test_categorical, verbose=0)

    # Readout layer weights
    weights = model.get_weights()

    # Save model if specified
    if save_model and standardize:
        model.save(os.path.join(out_dir_3, 'model_first_run_std.keras'))
    elif save_model and not standardize:
        model.save(os.path.join(out_dir_3, 'model_first_run_non_std.keras'))

    return final_loss, final_accuracy, weights, pattern_accuracies_evo, losses_evo

def run_multiple_simulations(num_runs, standardize=True, include_bias=True):
    """
    Run multiple simulations and collect results.

    Parameters:
    - num_runs (int): Number of simulation runs.
    - standardize (bool): Whether to standardize the datasets.
    - include_bias (bool): Whether to include a bias voltage in the dataset.

    Returns:
    - final_losses_all_runs (list): List of final loss values for each run.
    - final_accuracies_all_runs (list): List of final accuracy values for each run.
    - weights_all_runs (list): List of model weights for each run.
    - losses_evo_all_runs (list): Training and validation loss values over epochs for each run.
    - pattern_accuracies_evo_all_runs (list): Accuracy evolution per pattern across all runs.
    """
    if standardize: 
        print("Simulations with standardized data:")
    else: 
        print("Simulations without standardized data:")

    # Initialize lists to store results from all runs
    final_losses_all_runs = []
    final_accuracies_all_runs = []
    weights_all_runs = []
    losses_evo_all_runs = []
    pattern_accuracies_evo_all_runs = []

    for i in range(num_runs):
        save_model = (i == 0) # Save the model only for the first run
        final_loss, final_accuracy, weights, pattern_accuracies_evo, losses_evo = run_single_simulation(standardize=standardize, save_model=save_model, include_bias=include_bias)
       
        # Collect simulation results
        final_losses_all_runs.append(final_loss)
        final_accuracies_all_runs.append(final_accuracy)
        weights_all_runs.append(weights)
        losses_evo_all_runs.append(losses_evo)
        pattern_accuracies_evo_all_runs.append(pattern_accuracies_evo)
        print(f"Run {i+1}/{num_runs} completed.")

    return final_losses_all_runs, final_accuracies_all_runs, weights_all_runs, losses_evo_all_runs, pattern_accuracies_evo_all_runs

# Number of simulation runs
num_runs = 2

# Run simulation with and without standardization
final_losses_all_runs_std, final_accuracies_all_runs_std, weights_all_runs_std, losses_evo_all_runs_std, pattern_accuracies_evo_all_runs_std = run_multiple_simulations(num_runs, standardize=True, include_bias=True)
final_losses_all_runs_non_std, final_accuracies_all_runs_non_std, weights_all_runs_non_std, losses_evo_all_runs_non_std, pattern_accuracies_evo_all_runs_non_std = run_multiple_simulations(num_runs, standardize=False, include_bias=True)

# Plot average accuracy and loss across all runs for standarized and non-stanardized data
plot_accuracy(out_dir_1, final_accuracies_all_runs_std, final_accuracies_all_runs_non_std)
plot_loss(out_dir_1, final_losses_all_runs_std, final_losses_all_runs_non_std)

# Plot accuracy histogram over multiple runs
plot_accuracy_histogram(out_dir_1, num_runs, final_accuracies_all_runs_std, final_accuracies_all_runs_non_std)

# Plot conductance maps for the first run in standarized and non-standardized cases
plot_conductance_map(out_dir_2, weights_all_runs_std[0][0], case_name="Standarized")
plot_conductance_map(out_dir_2, weights_all_runs_non_std[0][0], case_name="Non-Standarized")

# Plot validation accuracy evolution across all runs
plot_validation_accuracies(out_dir_1, pattern_accuracies_evo_all_runs_std, case_name="Standarized")
plot_validation_accuracies(out_dir_1, pattern_accuracies_evo_all_runs_non_std, case_name="Non-Standarized")

# Plot training and validation cross-entropy loss evolution across all runs
plot_training_validation_loss(out_dir_1, losses_evo_all_runs_std, case_name="Standarized")
plot_training_validation_loss(out_dir_1, losses_evo_all_runs_non_std, case_name="Non-Standarized")

# Load the first run model for standardized case and generate a confusion matrix for results on a completely new dataset
model_first_run = models.load_model(os.path.join(out_dir_3, 'model_first_run_std.keras'))
num_new_validation_datasets = 25
new_validation_datasets = generate_datasets(output_voltage, num_new_validation_datasets)
new_validation_datasets_standardized, _, _, _ = standardize_data(new_validation_datasets, new_validation_datasets)
new_validation_datasets_flat = new_validation_datasets_standardized.reshape(-1, num_output_voltages)
new_validation_datasets_with_bias = np.concatenate([new_validation_datasets_flat, V_bias * np.ones((new_validation_datasets_flat.shape[0], 1))], axis=1)

y_new_validation = np.tile(np.arange(4), num_new_validation_datasets)
predictions = model_first_run.predict(new_validation_datasets_with_bias, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

# Generate and plot the confusion matrix for the standardized case
conf_matrix = confusion_matrix(y_new_validation, predicted_labels, normalize='true')
plot_confusion_matrix(out_dir_3, conf_matrix, 4, case_name="Standarized")

# Load the first run model for non-standardized case and generate a confusion matrix for results on a completely new dataset
model_first_run_non_std = models.load_model(os.path.join(out_dir_3, 'model_first_run_non_std.keras'))
num_new_validation_datasets_non_std = 25
new_validation_datasets_non_std = generate_datasets(output_voltage, num_new_validation_datasets_non_std)
new_validation_datasets_flat_non_std = new_validation_datasets_non_std.reshape(-1, num_output_voltages)
new_validation_datasets_with_bias_non_std = np.concatenate([new_validation_datasets_flat_non_std, V_bias * np.ones((new_validation_datasets_flat_non_std.shape[0], 1))], axis=1)

y_new_validation_non_std = np.tile(np.arange(4), num_new_validation_datasets_non_std)
predictions_non_std = model_first_run_non_std.predict(new_validation_datasets_with_bias_non_std, verbose=0)
predicted_labels_non_std = np.argmax(predictions_non_std, axis=1)

# Generate and plot the confusion matrix for the non-standardized case
conf_matrix = confusion_matrix(y_new_validation_non_std, predicted_labels_non_std, normalize='true')
plot_confusion_matrix(out_dir_3, conf_matrix, 4, case_name="Non-Standarized")