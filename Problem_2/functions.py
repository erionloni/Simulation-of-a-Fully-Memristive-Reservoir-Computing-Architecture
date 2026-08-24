import os
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sys import exit

#%% GRAPH DIMENSION (row and column lists)
def generate_row_col_lists(num_rows, num_cols):
    """
    Generate lists of row and column identifiers for the crossbar array.

    Parameters:
    - num_rows (int): Number of rows in the crossbar.
    - num_cols (int): Number of columns in the crossbar.

    Returns:
    - tuple: Two lists containing row and column identifiers, respectively.
    """
    row_list = [f'R{i}' for i in range(num_rows)]
    col_list = [f'C{i}' for i in range(num_cols)]
    return row_list, col_list

#%% GRAPH DEFINITION (reservoir)
def define_crossbar_graph(num_rows, num_cols):
    """
    Define a crossbar array graph representing the reservoir structure.

    Parameters:
    - num_rows (int): Number of rows in the crossbar.
    - num_cols (int): Number of columns in the crossbar.

    Returns:
    - networkx.DiGraph: A directed graph representing the crossbar array.
    """
    G = nx.DiGraph()

    row_lines = [f'R{i}' for i in range(num_rows)]
    col_lines = [f'C{i}' for i in range(num_cols)]
    
    # Add nodes to the graph
    for node in row_lines + col_lines:
        G.add_node(node)

    # Add edges between row nodes and column nodes
    for r in row_lines:
        for c in col_lines:
            G.add_edge(r, c)

    return G

#%% GRAPH INITIALIZATION (reservoir)  
def initialize_crossbar_attributes(G, Yin):
    """
    Initialize the attributes of the crossbar array graph, including edge admittance and node voltages.

    Parameters:
    - G (networkx.DiGraph): The crossbar graph.
    - Yin (float): Initial admittance value for the edges.

    Returns:
    - networkx.DiGraph: The graph with initialized attributes.
    """
    for u, v in G.edges():
        G[u][v]['Y'] = Yin  # Initialize edge admittance
        G[u][v]['deltaV'] = 0  # Initialize voltage difference across edges
        G[u][v]['g'] = 0  # Initialize normalized conductance

    for n in G.nodes():
        G.nodes[n]['V'] = 0  # Initialize node voltages
 
    return G

#%% VISUALIZATIONS (reservoir) 
def plot_directed_graph(G, num_rows, num_cols, pulse_amplitude, g_min, g_max, save_path=None, filename_prefix=None):
    """
    Visualize the crossbar array graph with node voltages and edge conductance. Optionally save the visualization.

    Parameters:
    - G (networkx.DiGraph): The crossbar graph.
    - num_rows (int): Number of rows in the crossbar.
    - num_cols (int): Number of columns in the crossbar.
    - pulse_amplitude (float): Maximum pulse amplitude for color scaling.
    - g_min (float): Minimum conductance value for edge color scaling.
    - g_max (float): Maximum conductance value for edge color scaling.
    - save_path (str, optional): Directory to save the image. Default is None.
    - filename_prefix (str, optional): Prefix for the filename. Default is None.
    """
    # Set node positions for visualization
    pos = {}
    row_lines = [f'R{i}' for i in range(num_rows)]
    col_lines = [f'C{i}' for i in range(num_cols)]

    for i, r in enumerate(row_lines):
        pos[r] = (0, num_rows - i)
    for i, c in enumerate(col_lines):
        pos[c] = (1, num_rows - i)
    
    # Plot the graph
    plt.figure(figsize=(8, 8))
    nx.draw_networkx(G, pos,
                     node_size=60,
                     node_color=[G.nodes[n]['V'] for n in G.nodes()],
                     cmap=plt.cm.Blues,
                     vmin=0,
                     vmax=pulse_amplitude + pulse_amplitude,
                     width=4,
                     edge_color=[G[u][v]['Y'] for u, v in G.edges()],
                     edge_cmap=plt.cm.Reds,
                     edge_vmin=g_min,
                     edge_vmax=g_max,
                     with_labels=True,
                     font_size=6)

    # Save the graph visualization if a save path is provided
    if save_path and filename_prefix:
        filename = f"{filename_prefix}_directed_graph.png"
        plt.savefig(os.path.join(save_path, filename))
    plt.close()

def plot_conductance_matrix(G, num_rows, num_cols, pulse_amplitude, g_min, g_max, save_path=None, filename_prefix=None, vmin=None, vmax=None):
    """
    Plot the conductance matrix of the crossbar array. Optionally save the visualization.

    Parameters:
    - G (networkx.DiGraph): The crossbar graph.
    - num_rows (int): Number of rows in the crossbar.
    - num_cols (int): Number of columns in the crossbar.
    - pulse_amplitude (float): Pulse amplitude for scaling.
    - g_min (float): Minimum conductance value for edge color scaling.
    - g_max (float): Maximum conductance value for edge color scaling.
    - save_path (str, optional): Directory to save the image. Default is None.
    - filename_prefix (str, optional): Prefix for the filename. Default is None.
    - vmin (float, optional): Minimum value for color scaling. Default is None.
    - vmax (float, optional): Maximum value for color scaling. Default is None.
    """
    # Initialize the conductance matrix
    conductance_matrix = np.zeros((num_rows, num_cols))
    for u, v, data in G.edges(data=True):
        row = int(u[1:])
        col = int(v[1:])
        conductance_matrix[row, col] = data['Y']

    # Plot the conductance matrix
    plt.figure(figsize=(6, 6))
    plt.imshow(conductance_matrix, cmap='Blues', interpolation='none', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Conductance [S]')

    # Set the ticks to integer values only
    plt.xticks(ticks=np.arange(num_cols), labels=np.arange(1, num_cols + 1))
    plt.yticks(ticks=np.arange(num_rows), labels=np.arange(1, num_rows + 1))
    
    # Save the plot if a save path is provided
    if save_path and filename_prefix:
        filename = f"{filename_prefix}_conductance_matrix.png"
        plt.savefig(os.path.join(save_path, filename))
    plt.close()

def plot_output_voltages_FOM(output_voltage, num_cols, digit_labels, save_path=None):
    """
    Plot reservoir output voltages for the FOM. Optionally save the visualization.

    Parameters:
    - output_voltage (list): List of output voltages for each pattern.
    - num_cols (int): Number of columns in the crossbar.
    - digit_labels (list): List of digit labels (e.g., 0, 1, 2, 3).
    - save_path (str, optional): Directory to save the plot.
    """
    num_patterns = len(output_voltage)
    fig, axes = plt.subplots(1, num_patterns, figsize=(20, 5), sharey=True)

    # Ensure axes is always iterable
    if num_patterns == 1:
        axes = [axes]

    # Plot each pattern's output voltage as a bar chart
    for i, ax in enumerate(axes):
        ax.bar(range(1, num_cols + 1), output_voltage[i], color='lightblue', edgecolor='black')
        ax.set_xlabel('Reservoir output')
        ax.set_title(f'Digit {digit_labels[i]}')  # Set the title to display the digit
        ax.set_xticks(range(1, num_cols + 1))
        ax.set_ylim(0, max(map(max, output_voltage)) + 0.01)

    axes[0].set_ylabel('Output voltage (V)')

    # Save the bar plots
    if save_path:
        plot_filename = os.path.join(save_path, 'output_voltages.png')
        plt.savefig(plot_filename)
    plt.close()

def plot_output_voltages_COM(organized_output_voltage, num_cols, pattern_name, read_out_points, save_path=None):
    """
    Plot reservoir output voltages for the COM. Optionally save the visualization.

    Parameters:
    - organized_output_voltage (dict): Dictionary of output voltages organized by pattern and read point.
    - num_cols (int): Number of columns in the crossbar.
    - pattern_name (list): List of pattern names.
    - read_out_points (list): List of read out points to plot.
    - out_dir (str): Directory to save the plot.
    """
    # Create subplots
    fig, axes = plt.subplots(len(read_out_points), len(pattern_name), figsize=(20, 5 * len(read_out_points)), sharey=True)

    # Plot each pattern in separate subplots for each read point
    for i, digit in enumerate(organized_output_voltage.keys()):
        for j, read_point in enumerate(organized_output_voltage[digit].keys()):
            ax = axes[j, i] if len(read_out_points) > 1 else axes[i]
            voltages = organized_output_voltage[digit][read_point]
            ax.bar(range(1, num_cols + 1), voltages, color='lightblue', edgecolor='black')
            ax.set_xlabel('Reservoir output')
            ax.set_title(f'{pattern_name[digit]} (t={read_point})')
            ax.set_xticks(range(1, num_cols + 1))
            ax.set_ylim(0, max(map(lambda v: max(v), [voltages for pattern in organized_output_voltage.values() for voltages in pattern.values()])) + 0.01)

    axes[0, 0].set_ylabel('Output voltage (V)')

    # Save the bar plots
    if save_path:
        plot_filename = os.path.join(save_path, 'output_voltages.png')
        plt.savefig(plot_filename)
    plt.close()

def plot_conductance_evolution(conductance_evolution, G, save_path=None, vmin=None, vmax=None):
    """
    Plot conductance evolution over all time steps during the writing phase. Optionally save the visualization.

    Parameters:
    - conductance_evolution (list): List of conductance snapshots over time.
    - G (networkx.DiGraph): The crossbar graph.
    - out_dir (str): Directory to save the plot.
    - vmin (float, optional): Minimum value for color scaling. Default is None.
    - vmax (float, optional): Maximum value for color scaling. Default is None.
    """
    plt.figure(figsize=(25, 10))

    # Plot conductance evolution for each edge in the crossbar
    for i, (u, v) in enumerate(G.edges()):
        try:
            conductances = [snapshot[u][v]['Y'] for snapshot in conductance_evolution]
            plt.plot(range(1, len(conductances) + 1), conductances, label=f'Edge ({u},{v})', marker='o', markersize=4, linewidth=1)
        except Exception as e:
            print(f"Error processing edge ({u},{v}):", e)

    plt.title('Conductance Evolution Over All Time Steps During Writing Phase')
    plt.xlabel('Time step')
    plt.ylabel('Conductance (S)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')
    plt.grid(True)

    # Save the conductance evolution plot
    if save_path:
        plot_filename = os.path.join(save_path, 'conductance_evolution.png')
        plt.savefig(plot_filename)
    plt.close()

#%% VOLTAGE ANALYSIS (reservoir)  
def voltage_analysis(G, V_list, V_list_inv, row_list, col_list):
    """
    Perform voltage node analysis for crossbar arrays. Applies original voltages to row lines 
    and inverted voltages to column lines.

    Parameters:
    - G (networkx.DiGraph): The crossbar graph.
    - V_list (list): List of input voltages for row lines.
    - V_list_inv (list): List of inverted input voltages for column lines.
    - row_list (list): List of row node identifiers.
    - col_list (list): List of column node identifiers.

    Returns:
    - networkx.DiGraph: The updated crossbar graph with node voltages and edge currents.
    """
    # Error checking for voltage inputs
    if 'f' in V_list or 'f' in V_list_inv:
        print(f"Error: Faulty voltage 'f' detected. Simulation stopped.")
        exit(1)

    if len(V_list) != len(row_list):
        print('Error: Input Voltage list and row node list must be equal in length!')
        exit(1)

    if len(V_list_inv) != len(col_list):
        print('Error: Inverted Voltage list and column node list must be equal in length!')
        exit(1)

    # Apply voltages to the nodes in the graph
    for i, r in enumerate(row_list):
        G.nodes[r]['V'] = V_list[i]
    for i, c in enumerate(col_list):
        G.nodes[c]['V'] = V_list_inv[i]

    # Update edge voltage differences and currents
    for u, v in G.edges():
        G[u][v]['deltaV'] = G.nodes[u]['V'] - G.nodes[v]['V']
        G[u][v]['I'] = G[u][v]['deltaV'] * G[u][v]['Y']
        G[u][v]['Irounded'] = np.round(G[u][v]['I'], 2)

    return G

#%% DATASET TO PULSE CONVERSION  
def dataset_to_pulse_FOM(digit_rows, digit_cols, pulse_timesteps, idle_timesteps, digit_list, input_digit):
    """
    Convert a dataset of digit patterns to a sequence of pulses.

    Parameters:
    - digit_rows (int): Number of rows in the digit pattern.
    - digit_cols (int): Number of columns in the digit pattern.
    - pulse_timesteps (int): Number of timesteps for each pulse.
    - idle_timesteps (int): Number of idle timesteps between pulses.
    - digit_list (list): List of digit patterns.
    - input_digit (int): Index of the digit pattern to convert.

    Returns:
    - list: A list representing the pulse sequence for the input digit pattern.
    """
    train_pulse = [[[] for _ in range(digit_cols)] for _ in range(digit_rows)]

    bit_0 = [0] * (idle_timesteps + pulse_timesteps + idle_timesteps)
    bit_1 = [0] * idle_timesteps + [1] * pulse_timesteps + [0] * idle_timesteps

    # Generate pulses for each cell in the digit pattern
    for i in range(digit_rows):
        for j in range(digit_cols):
            digit_cell = digit_list[input_digit][i][j]
            cell_value = int(digit_cell == 1)
            train_pulse[i][j] = bit_0 * (1 - cell_value) + bit_1 * cell_value
        train_pulse[i] = [element for item in train_pulse[i] for element in item]  # Flatten the list

    return train_pulse

def dataset_to_pulse_COM(digit_rows, digit_cols, pulse_timesteps, idle_timesteps, read_timesteps, digit_list, input_digit):
    """
    Convert a dataset of digit patterns to a sequence of pulses, including the read phase.

    Parameters:
    - digit_rows (int): Number of rows in the digit pattern.
    - digit_cols (int): Number of columns in the digit pattern.
    - pulse_timesteps (int): Number of timesteps for each pulse.
    - idle_timesteps (int): Number of idle timesteps between pulses.
    - read_timesteps (int): Number of timesteps in the read phase.
    - digit_list (list): List of digit patterns.
    - input_digit (int): Index of the digit pattern to convert.

    Returns:
    - list: A list representing the pulse sequence for the input digit pattern, including the read phase.
    """
    train_pulse = [[[] for _ in range(digit_cols)] for _ in range(digit_rows)]

    bit_0 = [0] * (idle_timesteps + pulse_timesteps)
    bit_1 = [0] * idle_timesteps + [1] * pulse_timesteps

    # Generate pulses for each cell in the digit pattern
    for i in range(digit_rows):
        for j in range(digit_cols):
            digit_cell = digit_list[input_digit][i][j]
            cell_value = int(digit_cell == 1)
            train_pulse[i][j] = bit_0 * (1 - cell_value) + bit_1 * cell_value
        train_pulse[i] = [element for item in train_pulse[i] for element in item]  # Flatten the list

    # Append the final read phase
    final_read = [0] * read_timesteps
    for i in range(digit_rows):
        train_pulse[i].extend(final_read)

    return train_pulse

#%% UPDATE EDGE WEIGHT (Miranda's model)  
def update_edge_weigths(G, delta_t, Y_min, Y_max, kp0, eta_p, kd0, eta_d, variability_level=0.0):
    """
    Update edge weights (conductance) based on Miranda's model, including variability.

    Parameters:
    - G (networkx.DiGraph): The crossbar graph.
    - delta_t (float): Time step.
    - Y_min (float): Minimum conductance.
    - Y_max (float): Maximum conductance.
    - kp0 (float): Rate constant for potentiation.
    - eta_p (float): Nonlinearity parameter for potentiation.
    - kd0 (float): Rate constant for depression.
    - eta_d (float): Nonlinearity parameter for depression.
    - variability_level (float, optional): Noise level to apply to the conductance value. Default is 0.0.

    Returns:
    - networkx.DiGraph: The updated crossbar graph with modified edge weights.
    """
    for u, v in G.edges():
        # Calculate the voltage difference across the edge
        G[u][v]['deltaV'] = abs(G.nodes[u]['V'] - G.nodes[v]['V'])
    
        # Calculate potentiation and depression rates
        G[u][v]['kp'] = kp0 * math.exp(eta_p * G[u][v]['deltaV'])
        G[u][v]['kd'] = kd0 * math.exp(-eta_d * G[u][v]['deltaV'])
        
        # Update the conductance based on Miranda's model
        G[u][v]['g'] = (G[u][v]['kp'] / (G[u][v]['kp'] + G[u][v]['kd'])) * \
                        (1 - (1 - (1 + (G[u][v]['kd'] / G[u][v]['kp']) * G[u][v]['g'])) * \
                        math.exp(-(G[u][v]['kp'] + G[u][v]['kd']) * delta_t))
    
        G[u][v]['Y'] = Y_min * (1 - G[u][v]['g']) + Y_max * G[u][v]['g']
        
        # Add variability to the conductance value
        variability = np.random.normal(0, variability_level)  
        G[u][v]['Y'] += variability  

        # Ensure conductance remains within bounds
        G[u][v]['Y'] = max(Y_min, min(G[u][v]['Y'], Y_max))
     
    return G
