import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.interpolate import interp1d

# Use LaTeX for rendering the text.
plt.rc('text', usetex=True)
plt.rc('font', family='serif')  # Optional: to use a serif font
plt.rc('text.latex', preamble=r'\usepackage{amsmath}') 

# Setup dict for the names of the graphs
names = [r'$\log|\mathcal{H}|$', r"$\mathcal{H}$", r"$\varphi$", r"$K$", r"$K_{\varphi}$", r"$\gamma_{rr}$", r"$\alpha$"] 
name_dict = {i+3: name for i, name in enumerate(names)}

def find_filelist(out_directory, r_resolution):
    """
    Find all files in the output directory that match the specified resolution.
    """
    return sorted(glob.glob(os.path.join(out_directory, f'out{r_resolution}-*.txt')))

def find_dt(out_directory, r_resolution):
    _, dt = np.loadtxt(os.path.join(out_directory, f"t_dt_{r_resolution}.txt"))
    return dt


def select_files(file_list, N, first = False):
    """
    Select N files from the list of files. If N > total, selects all. 
    Includes first and last, then evenly spaced in between based on the timestep.
    """
    if first:
        return file_list[:N]

    if N >= len(file_list):
        # If N is greater than or equal to the total number of files, return all files
        return file_list

    if N == 1:
        # If N is 1, return only the last file
        return [file_list[-1]]

    indices = np.linspace(0, len(file_list) - 1, N, dtype=int)
    indices[0] = 0
    indices[-1] = len(file_list) - 1

    # Select the files based on the computed indices
    file_list = [file_list[i] for i in indices]
    return file_list

def get_r_T_from_raw(data_raw):
    """
    Extracts unique r and tensors from the raw data (given in cartesian)    
    """
    x = data_raw[:, 0]
    y = data_raw[:, 1]
    z = data_raw[:, 2]
    r_raw = np.sqrt(x**2 + y**2 + z**2)

    # For a spherically symmetric problem, the x,y,z values resulting
    # in the same r hold the same data. 
    r, ind = np.unique(r_raw, return_index=True)
    data = data_raw[ind]
    return r, data

def get_data_from_file(tensor_ind, file):
    try:
        data_raw = np.loadtxt(file)
    except FileNotFoundError:
        print(f"File {file} not found.")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
    r, data = get_r_T_from_raw(data_raw)
    return r, data[:, tensor_ind]

def get_all_from_filelist(tensor_ind, file_list):
    data = []
    r = None
    for file in file_list:
        try:
            r, T = get_data_from_file(tensor_ind, file)
        except Exception as e:
            print(f"Error for file {file}: {e}")
            break
        data.append(T)
    return r, data

def get_file_index(file):
    return os.path.basename(file).split('-')[-1].split('.')[0].zfill(4)


# Plot functions (with the help of ChatGPT)
def plot_evolution(tensor_ind, out_directory, r_resolutions, N_plots=6, name_dict=name_dict, title =None, save_at = None, ylim = None):
    if not isinstance(r_resolutions, list):
        r_resolutions = [r_resolutions]

    num_cols = 2  # Number of columns in the grid
    num_rows = (N_plots + num_cols - 1) // num_cols  # Calculate the number of rows needed

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 6))
    axs = axs.flatten()  # Flatten the 2D array of axes to 1D for easier indexing
    colors = ['purple', 'orange', 'c']
    for j,r_resolution in enumerate(r_resolutions):
        file_list = find_filelist(out_directory, r_resolution)
        dt = find_dt(out_directory, r_resolution)

        selected_files = select_files(file_list, N_plots)

        for i, file in enumerate(selected_files):
            # Get data for each set
            r, T = get_data_from_file(tensor_ind, file)
            file_index = get_file_index(file)

            # Plot sGB_ADM with a bold, solid line
            axs[i].plot(r, T, label=f'$N_r = $ {r_resolution}', color = colors[j], linewidth=2)

            # Plot Analytical with a semi-transparent line
            # Plot limits and decorations
            axs[i].set_xscale('log')
            axs[i].set_xlabel('$r/M$', fontsize=17)
            axs[i].set_ylabel(name_dict[tensor_ind], fontsize=17)
            axs[i].set_title(f'$t/M \\approx {round(int(file_index) * dt + dt, 2)}$', fontsize=17)

            # Improved legend placement
            axs[i].legend(fontsize=14)

            # Add a grid for better readability
            axs[i].grid(alpha=0.4)

            axs[i].tick_params(axis='both', which='major', labelsize=14)
    for i in range(len(axs)):
        axs[i].axvline(x=2, color='black', linestyle='--', linewidth=1.5, label='$r = 2M$', alpha=0.5)
    if ylim is not None:
        for i in range(len(axs)):
            axs[i].set_ylim(ylim)
    # Add a title to the whole plot
    title = title
    if title is None:
        title = f"{name_dict[tensor_ind]} during evolution"
    fig.suptitle(title, fontsize=25)

    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for the title
    if save_at is not None:
        plt.savefig(save_at,dpi=450)
    plt.show()

def plot_evolution_w_anal(tensor_ind, a_sol, out_directory, r_resolutions, N_plots=6, name_dict=name_dict, title=None, save_at=None):
    if not isinstance(r_resolutions, list):
        r_resolutions = [r_resolutions]

    num_cols = 2  # Number of columns in the grid
    num_rows = (N_plots + num_cols - 1) // num_cols  # Calculate the number of rows needed

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 6))
    axs = axs.flatten()  # Flatten the 2D array of axes to 1D for easier indexing
    colors = ['purple', 'orange', 'c']
    r = []
    for j,r_resolution in enumerate(r_resolutions):
        file_list = find_filelist(out_directory, r_resolution)
        dt = find_dt(out_directory, r_resolution)

        selected_files = select_files(file_list, N_plots)

        for i, file in enumerate(selected_files):
            # Get data for each set
            r, T = get_data_from_file(tensor_ind, file)
            file_index = get_file_index(file)

            # Plot sGB_ADM with a bold, solid line
            axs[i].plot(r, T, label=f'$N_r = $ {r_resolution}', color = colors[j], linewidth=2)

            # Plot Analytical with a semi-transparent line

            # Plot limits and decorations
            axs[i].set_xscale('log')
            axs[i].set_xlabel('$r/M$', fontsize=17)
            axs[i].set_ylabel(name_dict[tensor_ind], fontsize=17)
            axs[i].set_title(f'$t/M \\approx {round(int(file_index) * dt, 2)}$', fontsize=17)

            # Improved legend placement
            # axs[i].legend(fontsize=14)

            # Add a grid for better readability
            axs[i].grid(alpha=0.4)

            axs[i].tick_params(axis='both', which='major', labelsize=14)

    anal = a_sol(r)
    for i in range(len(axs)):
        axs[i].axvline(x=2, color='black', linestyle='--', linewidth=1.5, label='$r = 2M$', alpha=0.5)
        axs[i].plot(r, anal, label='Analytical', color='black', linewidth=1.5, alpha=0.5)
        axs[i].legend(fontsize=14)  # Ensure the legend is updated with the new plots

    # Add a title to the whole plot
    title = title
    if title is None:
        title = f"{name_dict[tensor_ind]} during evolution"
    fig.suptitle(title, fontsize=25)

    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for the title

    if save_at is not None:
        plt.savefig(save_at,dpi=450)
    plt.show()

def plot_evolution_first(tensor_ind, out_directory, r_resolutions, N_plots=6, name_dict=name_dict, title =None, save_at = None, ylim = None, log = True, r_lim = None):
    if not isinstance(r_resolutions, list):
        r_resolutions = [r_resolutions]

    num_cols = 2  # Number of columns in the grid
    num_rows = (N_plots + num_cols - 1) // num_cols  # Calculate the number of rows needed

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 6))
    axs = axs.flatten()  # Flatten the 2D array of axes to 1D for easier indexing
    colors = ['purple', 'orange', 'c']
    for j,r_resolution in enumerate(r_resolutions):
        file_list = find_filelist(out_directory, r_resolution)
        dt = find_dt(out_directory, r_resolution)

        selected_files = select_files(file_list, N_plots, first=True)

        for i, file in enumerate(selected_files):
            # Get data for each set
            r, T = get_data_from_file(tensor_ind, file)
            file_index = get_file_index(file)

            # Plot sGB_ADM with a bold, solid line
            axs[i].plot(r, T, label=f'$N_r = $ {r_resolution}', color = colors[j], linewidth=2)

            # Plot Analytical with a semi-transparent line
            # Plot limits and decorations
            if log:
                axs[i].set_xscale('log')
            axs[i].set_xlabel('$r/M$', fontsize=17)
            axs[i].set_ylabel(name_dict[tensor_ind], fontsize=17)
            axs[i].set_title(f'$t/M \\approx {round(int(file_index) * dt, 4)}$', fontsize=17)

            # Improved legend placement
            axs[i].legend(fontsize=14)

            # Add a grid for better readability
            axs[i].grid(alpha=0.4)

            axs[i].tick_params(axis='both', which='major', labelsize=14)
    for i in range(len(axs)):
        axs[i].axvline(x=2, color='black', linestyle='--', linewidth=1.5, label='$r = 2M$', alpha=0.5)
    if ylim is not None:
        for i in range(len(axs)):
            axs[i].set_ylim(ylim)
    if r_lim is not None:
        for i in range(len(axs)):
            axs[i].set_xlim(0,r_lim)
    # Add a title to the whole plot
    title = title
    if title is None:
        title = f"{name_dict[tensor_ind]} during evolution"
    fig.suptitle(title, fontsize=25)

    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for the title
    if save_at is not None:
        plt.savefig(save_at,dpi=200)
    plt.show()

def plot_interpolate_rescale(tensor_ind, out_directory, r_resolutions, N_plots=6, name_dict=name_dict, title =None, save_at = None, ylim = None, log = True, r_lim = None, first = False):
    files = [find_filelist(out_directory, res) for res in r_resolutions]
    for i in range(len(files)):
        files[i] = select_files(files[i], N_plots, first = first)
    files = list(map(list, zip(*files)))

    num_cols = 2  # Number of columns in the grid
    num_rows = (N_plots + num_cols - 1) // num_cols  # Calculate the number of rows needed

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 6))
    axs = axs.flatten()  # Flatten the 2D array of axes to 1D for easier indexing

    colors = ['purple', 'orange', 'c']
    for i,file_at_step in enumerate(files):
        rs = []	
        hs = []	
        for res, f in zip(r_resolutions,file_at_step):
            rr, hh = get_data_from_file(3, f)
            hh = hh + np.log10((res/r_resolutions[0])**4)
            rs.append(rr)
            hs.append(hh)
        r1, r2, r3 = rs
        H1, H2, H3 = hs
        # Create an interpolation function for H200 on r200 grid
        interp_func1 = interp1d(r2, H2, kind='linear', fill_value="extrapolate")
        interp_func2 = interp1d(r3, H3, kind='linear', fill_value="extrapolate")

        # Interpolate H200 values on the r100 grid
        H2_on_r1 = interp_func1(r1)
        H3_on_r1 = interp_func2(r1)


        axs[i].plot(r1, H1, label=f'$N_r = $ {r_resolutions[0]}', color=colors[0], linewidth=2)
        axs[i].plot(r1, H2_on_r1, label=f'$N_r = $ {r_resolutions[1]}', color=colors[1], linewidth=2)
        axs[i].plot(r1, H3_on_r1, label=f'$N_r = $ {r_resolutions[2]}', color=colors[2], linewidth=2)

        axs[i].set_xscale('log')
        axs[i].axvline(x=2, color='black', linestyle='--', linewidth=1.5, label='$r = 2M$', alpha = 0.5)
        axs[i].axvline(x = 100)
        axs[i].axvline(x = 1/100)
        # axs[i].set_ylim(-15, 2.5)
        axs[i].set_xlabel('$r/M$', fontsize=17)
        axs[i].set_ylabel(r'$\log_{10}\big[|\mathcal{H}| (N_{r}/180)^4$\big]', fontsize=17)
        dt = find_dt(out_directory,r_resolutions[0])
        file_index = get_file_index(file_at_step[0])
        axs[i].set_title(f'$t/M \\approx {round(int(file_index) * dt, 4)}$', fontsize=17)

        # Improved legend placement
        axs[i].legend(fontsize = 14)

        # Add a grid for better readability
        axs[i].grid(alpha=0.4)

        axs[i].tick_params(axis='both', which='major', labelsize=14)
    if ylim is not None:
        for i in range(len(axs)):
            axs[i].set_ylim(ylim)
    if r_lim is not None:
        for i in range(len(axs)):
            axs[i].set_xlim(0,r_lim)
    # Add a title to the whole plot
    if title is None:
        title = f"{name_dict[tensor_ind]} during evolution"
    fig.suptitle(title, fontsize=25)

    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for the title
    if save_at is not None:
        plt.savefig(save_at,dpi=200)
    plt.show()