import os
import numpy as np
from functions import (generate_row_col_lists, define_crossbar_graph,
                       initialize_crossbar_attributes, plot_directed_graph, dataset_to_pulse_FOM,
                       voltage_analysis, update_edge_weigths, plot_conductance_matrix, 
                       plot_output_voltages_FOM, plot_conductance_evolution)

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
out_dir_1 = './Problem_1/FOM Outputs/Reservoir/Conductance Evolution/'  # Directory for reservoir conductance evolution plots
out_dir_2 = './Problem_1/FOM Outputs/Reservoir/Reservoir Final states/'  # Directory for reservoir final state plots
out_dir_3 = './Problem_1/FOM Outputs/Reservoir/Voltage Outputs/'  # Directory for reservoir output voltage plots

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
idle_time = 5e-4 - delta  # Idle time between pulses
read_time = 5.5e-3  #  Read phase duration

# Timesteps for different phases
read_timesteps = int(read_time / delta_read)  # Number of reading phase timesteps 
pulse_timesteps = int(pulse_time / delta_pot) + 1  # Number of pulse phase timesteps 
idle_timesteps = int(idle_time / delta_dep) + 1  # Number of idle phase timesteps 

one_pulse = 2*idle_timesteps + pulse_timesteps # Single pulse cycle timesteps

# Time arrays of a single pulse cycle
time_write_1 = np.linspace(0, idle_time, idle_timesteps) 
time_write_2 = np.linspace(idle_time + delta, idle_time + delta + pulse_time, pulse_timesteps) 
time_write_3 = np.linspace(idle_time + pulse_time + 2*delta, idle_time + pulse_time + 2*delta + idle_time, idle_timesteps) 

time_write_tot = np.append(np.append(time_write_1, time_write_2), time_write_3)
time_write = time_write_tot  

# Time array of entire pattern consisting of multiple columns
for i in range(1, digit_cols):
    time_write = np.append(time_write, time_write_tot + time_write[-1] + delta_dep)

timesteps_write = len(time_write) 

# List storing special timesteps
int_point = [[] for ip in range(digit_cols + 2)]
for ip in range(digit_cols + 2): 
    int_point[ip] = one_pulse * ip - idle_timesteps - 1
int_point[-1] = timesteps_write + read_timesteps - 1
int_point[0] = len(time_write_1) - 1  

# Explanation of what the special timesteps represent:
# int_point[0] - End of the first idle phase, where the first pulse begins
# int_point[1] to int_point[-2] - Timesteps right after each pulse
# int_point[-1] - The last point in time, marking the end of the simulation (after read phase)

#%% VOLTAGES

V_read = 100e-3  # Read voltage (in volts)
pulse_amplitude = 5  # Pulse voltage (in volts)

# Voltage arrays for each pattern
V_in_list = [[np.zeros((num_rows, timesteps_write))] for _ in range(num_digits_train)]
V_in_list_inv = [[np.zeros((num_cols, timesteps_write))] for _ in range(num_digits_train)]

# Convert pattern into pulse sequences 
train_pulses = [dataset_to_pulse_FOM(digit_rows, digit_cols, pulse_timesteps, idle_timesteps, digit_list_train, i) for i in range(num_digits_train)]
train_pulses_inv = [dataset_to_pulse_FOM(digit_rows, digit_cols, pulse_timesteps, idle_timesteps, digit_list_train_inv, i) for i in range(num_digits_train)]

# WL and BL signals
for digit in range(num_digits_train):  
    for t in range(timesteps_write):  
        # WL signals
        for node in range(num_rows):
            pulse_index = node % digit_rows  
            if t % one_pulse >= idle_timesteps and t % one_pulse < one_pulse - idle_timesteps:  
                V_in_list[digit][0][node][t] = train_pulses[digit][pulse_index][t] * pulse_amplitude + pulse_amplitude  
            else: 
                V_in_list[digit][0][node][t] = train_pulses[digit][pulse_index][t] * pulse_amplitude
        # BL signals
        for node in range(num_cols):
            pulse_index = node % digit_cols  
            V_in_list_inv[digit][0][node][t] = train_pulses_inv[digit][pulse_index][t] * pulse_amplitude

#%% CURRENTS AND CONDUCTANCES 

# Initialize arrays to store currents and conductances for each pattern
currents_by_pattern = {pattern: [] for pattern in range(num_digits_train)}  
conductances_by_pattern = {pattern: [] for pattern in range(num_digits_train)}  
currents_at_columns_by_pattern = []  

# Array to store conductance evolution
conductance_evolution = [[] for _ in range(num_digits_train)]

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

    # Writing phase
    for t in range(1, timesteps_write):
        delta_t = time_write[t] - time_write[t - 1]  # Time difference between timesteps
        G = update_edge_weigths(G, delta_t, g_min, g_max, kp0, eta_p, kd0, eta_d)  # Update edge weights

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

        # Store conductance evolution
        conductance_snapshot = np.array([[G[u][v]['Y'] for v in col_list] for u in row_list])
        conductance_evolution[digit].append(G.copy())

    # Reading phase
    for t in range(timesteps_write, timesteps_write + read_timesteps):
        delta_t = delta_read
        G = update_edge_weigths(G, delta_t, g_min, g_max, kp0, eta_p, kd0, eta_d)

        voltage_input = [V_read] * num_rows
        voltage_input_inv = [0] * num_cols

        G = voltage_analysis(G, voltage_input, voltage_input_inv, row_list, col_list)

    # Store final currents and conductances for each edge
    currents = {}
    conductances = {}
    for u, v in G.edges():
        currents[(u, v)] = G[u][v]['I']
        conductances[(u, v)] = G[u][v]['Y']
    currents_by_pattern[digit].append(currents)
    conductances_by_pattern[digit].append(conductances)

    # Plot final graph state for each pattern
    plot_directed_graph(G, num_rows, num_cols, pulse_amplitude, g_min, g_max, out_dir_2, f'pattern_{pattern_name[digit]}_final')
    plot_conductance_matrix(G, num_rows, num_cols, pulse_amplitude, g_min, g_max, out_dir_2, f'pattern_{pattern_name[digit]}_final', g_min, g_max)

    # Plot final state of first pattern (digit == 0); adjust for other patterns
    if digit == 0:
        plot_directed_graph(G, num_rows, num_cols, pulse_amplitude, g_min, g_max, out_dir_1, f'pattern_{pattern_name[digit]}_t5')
        plot_conductance_matrix(G, num_rows, num_cols, pulse_amplitude, g_min, g_max, out_dir_1, f'pattern_{pattern_name[digit]}_t_int5', g_min, g_max)
        
    # Calculate and store total currents for each column 
    total_currents_per_column = [0] * num_cols
    for (u, v), current in currents.items():
        if u in row_list and v in col_list:
            column_index = col_list.index(v)
            total_currents_per_column[column_index] += current
    currents_at_columns_by_pattern.append(total_currents_per_column)

#%% OUTPUT VOLTAGE CALCULATION AND PLOTTING

R_read = 100  # Read resistance in (in ohm)

# Output voltages
output_voltage = [[current * R_read for current in pattern] for pattern in currents_at_columns_by_pattern]
voltages = np.array(output_voltage).flatten()
pattern_labels = np.repeat(digit_train_class, len(output_voltage[0]))

# Bar plot of output voltages
plot_output_voltages_FOM(output_voltage, num_cols, pattern_name, out_dir_3)

#%% CONDUCTANCE EVOLUTION PLOTTING

# Conductance evolution of first pattern; adjust for other patterns
first_pattern_conductance_evolution = conductance_evolution[0]

# Plot conductance evolution 
plot_conductance_evolution(first_pattern_conductance_evolution, define_crossbar_graph(num_rows, num_cols), out_dir_1)