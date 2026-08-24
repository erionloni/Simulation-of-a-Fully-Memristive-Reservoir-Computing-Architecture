import os
import numpy as np
import matplotlib.pyplot as plt
from functions import (generate_row_col_lists, define_crossbar_graph, 
                       initialize_crossbar_attributes, plot_directed_graph, dataset_to_pulse_COM, 
                       voltage_analysis, update_edge_weigths, plot_conductance_matrix,
                       plot_output_voltages_COM, plot_conductance_evolution)

#%% RESERVOIR INITIALIZATION

# Memristor model constants
kp0 = 2.555173332603108574e-06  # Potentiation Rate Constant
kd0 = 6.488388862524891465e+01  # Depression Rate Constant
eta_p = 3.492155165334443012e+01  # Potentiation nonlinearity parameter  
eta_d = 5.590601016803570467e+00  # Depression nonlinearity parameter  
g_min = 0  # Memristor minimum conductance  
g_max = 2.723493729125820492e-03  # Memristor maximum conductance  
g0 = g_min  # Memristor initial conductance 

# Crossbar dimensions
num_rows = 4
num_cols = 4

# Row and column lists 
row_list, col_list = generate_row_col_lists(num_rows, num_cols)

#%% DATASET AND OUTPUT DIRECTORIES

# Output directories for different figures
out_dir_1 = './Problem_1/COM Outputs/Reservoir/Conductance Evolution/'  # Directory for reservoir conductance evolution plots
out_dir_2 = './Problem_1/COM Outputs/Reservoir/Reservoir Final states/'  # Directory for reservoir final state plots
out_dir_3 = './Problem_1/COM Outputs/Reservoir/Voltage Outputs/'  # Directory for reservoir output voltage plots

# Create directories if they do not exist
os.makedirs(out_dir_1, exist_ok=True)
os.makedirs(out_dir_2, exist_ok=True)
os.makedirs(out_dir_3, exist_ok=True)

# Load training data
file_to_train = './Problem_1/raw_data/pattern'  
file_train = file_to_train + '.txt'  # Contains training patterns in binary format
file_train_class = file_to_train + '_class.txt'  # Contains class labes for each training pattern

digit_train = np.loadtxt(file_train)  
digit_train_class = np.loadtxt(file_train_class)  

# Pattern names
pattern_name = ['diag1', 'diag2', 'horz', 'vert']  

# Digit dimensions and total rows storing information of all patterns
digit_rows = digit_cols = int((digit_train.shape[1]))
total_rows_train = int(len(digit_train))  

# Number of different patterns
num_digits_train = len(digit_train_class) 

# Training patterns and their inverses
digit_list_train = [digit_train[digit_rows * i : digit_rows * (i + 1)] for i in range(num_digits_train)] 
digit_list_train_inv = [1 - np.array(digit) for digit in digit_list_train]  # Invert each pattern

#%% TIME DEFINITIONS

# Time durations for different phases (in seconds)
delta_pot = 250e-6  # Time interval between two consecutive potentiation steps
delta_dep = 250e-6  # Time interval between two consecutive depression steps
delta_read = delta_dep  # Time interval between two consecutive read steps
delta = 250e-6  # Transition time between different phases

pulse_time = 10e-3 - delta  # Single pulse duration
read_time = 5.5e-3  #  Read phase duration
idle_time = 5e-4 - delta  # Idle time between pulses

# Timesteps for different phases
read_timesteps = int(read_time / delta_read)  # Number of reading phase timesteps 
pulse_timesteps = int(pulse_time / delta_pot) + 1  # Number of pulse phase timesteps 
idle_timesteps = int(idle_time / delta_dep) + 1  # Number of idle phase timesteps 

one_pulse = idle_timesteps + pulse_timesteps  # Single pulse cycle timesteps

# Time arrays of a single pulse cycle
time_write_1 = np.linspace(0, idle_time, idle_timesteps, endpoint=False)  
time_write_2 = np.linspace(idle_time + delta, idle_time + delta + pulse_time, pulse_timesteps, endpoint=False)  

time_write_tot = np.append(time_write_1, time_write_2)  
time_write = time_write_tot

# Time array of entire pattern consisting of multiple columns
for i in range(1, digit_cols):
    time_write = np.append(time_write, time_write_tot + time_write[-1] + delta_dep)

# Append final read phase
time_final_read = np.linspace(time_write[-1] + delta_dep, time_write[-1] + delta_dep + read_time, idle_timesteps, endpoint=False)
time_write = np.append(time_write, time_final_read)

timesteps_write = len(time_write) 

# List storing special timesteps
int_point = [0]  # First timestep
for i in range(1, digit_cols + 1):
    int_point.append(one_pulse * i - 1)
int_point.append(timesteps_write - 1)

# Explanation of what the special timesteps represent:
# int_point[0] - End of the first idle phase, where the first pulse begins
# int_point[1] to int_point[-2] - Timesteps right after each pulse
# int_point[-1] - The last point in time, marking the end of the simulation (after read phase)

# List storing read timesteps
read_out_points = []
for i in range(1, digit_cols + 1):
    read_out_points.append(one_pulse * i + idle_timesteps - 1)

#%% VOLTAGES

V_read = 100e-3  # Read voltage (in volts)
pulse_amplitude = 5  # Pulse voltage (in volts)

# Voltage arrays for each pattern
V_in_list = [[np.zeros((num_rows, timesteps_write))] for _ in range(num_digits_train)]
V_in_list_inv = [[np.zeros((num_cols, timesteps_write))] for _ in range(num_digits_train)]

# Convert pattern into pulse sequences 
train_pulses = [dataset_to_pulse_COM(digit_rows, digit_cols, pulse_timesteps, idle_timesteps, read_timesteps, digit_list_train, i) for i in range(num_digits_train)]
train_pulses_inv = [dataset_to_pulse_COM(digit_rows, digit_cols, pulse_timesteps, idle_timesteps, read_timesteps, digit_list_train_inv, i) for i in range(num_digits_train)]

# WL and BL signals
for digit in range(num_digits_train):
    for t in range(timesteps_write):
        # WL signals
        for node in range(num_rows):
            pulse_index = node % digit_rows  
            pulse_timestep = t % one_pulse
            if t < idle_timesteps:  
                V_in_list[digit][0][node][t] = 0
            elif pulse_timestep < idle_timesteps:  
                V_in_list[digit][0][node][t] = V_read
            elif pulse_timestep < (idle_timesteps + pulse_timesteps):  
                if train_pulses[digit][pulse_index][t] == 1:
                    V_in_list[digit][0][node][t] = 2 * pulse_amplitude
                else:
                    V_in_list[digit][0][node][t] = pulse_amplitude
            else:  
                V_in_list[digit][0][node][t] = 0
        # BL signals
        for node in range(num_cols):
            pulse_index = node % digit_cols  
            pulse_timestep = t % one_pulse
            if t < idle_timesteps:  
                V_in_list_inv[digit][0][node][t] = 0
            elif pulse_timestep < idle_timesteps:  
                V_in_list_inv[digit][0][node][t] = 0
            elif pulse_timestep < (idle_timesteps + pulse_timesteps):  
                if train_pulses_inv[digit][pulse_index][t] == 1:
                    V_in_list_inv[digit][0][node][t] = pulse_amplitude
                else:
                    V_in_list_inv[digit][0][node][t] = 0
            else:  
                V_in_list_inv[digit][0][node][t] = 0

#%% CURRENTS, CONDUCTANCES AND VOLTAGES
  
# Initialize arrays to store currents, conductances and voltages for each pattern
currents_by_pattern = {pattern: [] for pattern in range(num_digits_train)} 
conductances_by_pattern = {pattern: [] for pattern in range(num_digits_train)} 
currents_at_columns_by_pattern = []
output_voltage_by_read = []

# Array to store conductance evolution
conductance_evolution = [[] for _ in range(num_digits_train)]

R_read = 100  # Read resistance (in ohm)

#%% RESERVOIR SIMULATION

# Loop through each pattern
for digit in range(num_digits_train):
    print('Simulating pattern ' + pattern_name[digit])

    # Define crossbar graph and initialize attributes
    G = define_crossbar_graph(num_rows, num_cols)
    G = initialize_crossbar_attributes(G, g0)

    # Initial voltage inputs for first timestep
    voltage_input_0 = [V_in_list[digit][0][row][0] for row in range(num_rows)]
    voltage_input_inv_0 = [V_in_list_inv[digit][0][col][0] for col in range(num_cols)]

    # Perform voltage analysis
    G = voltage_analysis(G, voltage_input_0, voltage_input_inv_0, row_list, col_list)

    # Writing and reading phases
    for t in range(timesteps_write):
        delta_t = time_write[t] - time_write[t - 1] # Time difference between timesteps
        G = update_edge_weigths(G, delta_t, g_min, g_max, kp0, eta_p, kd0, eta_d) # Update edge weights
        
        # Update voltages for the current timestep
        voltage_input = [V_in_list[digit][0][row][t] for row in range(num_rows)]
        voltage_input_inv = [V_in_list_inv[digit][0][col][t] for col in range(num_cols)]

        # Perform voltage analysis on the graph for the current timestep
        G = voltage_analysis(G, voltage_input, voltage_input_inv, row_list, col_list)

        # Plot conductance evolution for the 1st pattern (digit == 0); adjust for other patterns
        if digit == 0 and t in int_point:
            filename_prefix = f'pattern_{pattern_name[digit]}_t{int_point.index(t)}'
            plot_directed_graph(G, num_rows, num_cols, pulse_amplitude, g_min, g_max, out_dir_1, filename_prefix)
            plot_conductance_matrix(G, num_rows, num_cols, pulse_amplitude, g_min, g_max, out_dir_1, filename_prefix, g_min, g_max)

        # Collect currents and calculate output voltages at readout points
        if t in read_out_points:
            currents = { (u, v): G[u][v]['I'] for u, v in G.edges() }
            conductances = { (u, v): G[u][v]['Y'] for u, v in G.edges() }
            currents_by_pattern[digit].append(currents)
            conductances_by_pattern[digit].append(conductances)

            # Calculate total current for each column node
            total_currents_per_column = [0] * num_cols
            for (u, v), current in currents.items():
                if u in row_list and v in col_list:
                    column_index = col_list.index(v)
                    total_currents_per_column[column_index] += current # contains the total current at each column node
            output_voltage_by_read.append((digit, t, [current * R_read for current in total_currents_per_column]))

        # Store the graph state
        conductance_evolution[digit].append(G.copy()) 

    # Plot final graph state for each pattern
    plot_directed_graph(G, num_rows, num_cols, pulse_amplitude, g_min, g_max, out_dir_2, f'pattern_{pattern_name[digit]}_final')
    plot_conductance_matrix(G, num_rows, num_cols, pulse_amplitude, g_min, g_max, out_dir_2, f'pattern_{pattern_name[digit]}_final', g_min, g_max)

#%% OUTPUT VOLTAGE CALCULATION AND PLOTTING

# Organize the output voltages by pattern and read point
organized_output_voltage = {digit: {read_point: [] for read_point in read_out_points} for digit in range(num_digits_train)}

for digit, t, voltages in output_voltage_by_read:
    organized_output_voltage[digit][t] = voltages

# Bar plot of output voltages
plot_output_voltages_COM(organized_output_voltage, num_cols, pattern_name, read_out_points, out_dir_3)

#%% CONDUCTANCE EVOLUTION PLOTTING

# Conductance evolution of first pattern; adjust for other patterns if necessary
first_pattern_conductance_evolution = conductance_evolution[0]

# Plot conductance evolution using the existing function
plot_conductance_evolution(first_pattern_conductance_evolution, define_crossbar_graph(num_rows, num_cols),out_dir_1)
