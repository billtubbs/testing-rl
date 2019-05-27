import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines.bench.monitor import load_results
from stable_baselines import results_plotter

log_filenames = {
    'learning': 'monitor.csv',
    'args': 'args.yaml',
    'best model': 'best_model.pkl',
    'final model': 'model.pkl',
    'log': 'log.out'
}

def survey_logs(results_dir, log_filenames=log_filenames,
                recursively=True, verbose=1):
    """Looks at all folders in directory results_dir
    to determine which contain log files."""

    if verbose > 1:
        print(f"\nLooking for log files in:\n '{results_dir}'")

    log_dirs = []
    log_dir_paths = []
    dir_contents = []

    for root, directories, filenames in os.walk(results_dir):

        # Go through all files except hidden ones
        selected_files = [f for f in filenames if not f.startswith('.')]
        checks = pd.Series(
            [f in selected_files for f in log_filenames.values()],
            index=log_filenames.keys()
        )
        if checks.sum() > 0:
            log_dirs.append(os.path.split(root)[-1])
            log_dir_paths.append(root)
            dir_contents.append(checks)
    index = pd.Index(log_dirs, name='Folder name')
    log_inventory = pd.DataFrame.from_records(dir_contents, index=index)
    log_inventory['Path'] = log_dir_paths

    if verbose > 0:
        print(f"{len(log_inventory)} log directories inventoried")

    return log_inventory

def combine_log_data(log_dirs, log_filenames=log_filenames,
                     check_paths=True, update_paths=False,
                     verbose=1):
    """
    """

    # First, go through log output (to check for errors)

    if verbose > 0:
        log_filename = log_filenames['log']

        if verbose > 1:
            print(f"\nChecking '{log_filename}' files...\n")
        for dir_path in log_dirs:
            dir_name = os.path.split(dir_path)[-1]
            filepath = os.path.join(dir_path, log_filename)
            try:
                with open(filepath) as f:
                    log_output = f.read()
            except FileNotFoundError:
                if verbose > 0:
                    print(f"{dir_name}: WARNING. No '{log_filename}'"
                          f" file found")
            else:
                if verbose > 1:
                    if log_output != '':
                        print(f"\nLog output for {dir_name}:")
                        print(log_output)
                    else:
                        print(f"{dir_name}: No output")

    # Read and combine arg values

    args_filename = log_filenames['args']
    if verbose > 1:
        print(f"\nChecking '{args_filename}' files...\n")

    args_df = []
    for dir_path in log_dirs:
        dir_name = os.path.split(dir_path)[-1]
        filepath = os.path.join(dir_path, args_filename)
        try:
            with open(filepath) as f:
                args = yaml.safe_load(f)
        except FileNotFoundError:
            if verbose > 0:
                print(f"{dir_name}: WARNING. No '{args_filename}' "
                      f"file in: {dir_path}")
        else:
            if verbose > 1:
                print(f"{dir_name}: {len(args)} args")

            # If the logs have been moved, updated the path argument
            if check_paths:
                if update_paths:
                    args['log_dir'] = dir_path
                else:
                    assert dir_path == args['log_dir'], "The 'log_dir' " \
                        f"path in '{args_filename}' does not match the " \
                        "actual file path.  Set update_paths=True to " \
                        "change these."

            args_df.append(pd.Series(args, name=dir_name))
    args_df = pd.DataFrame(args_df)
    duplicates = args_df.index.duplicated()
    assert not duplicates.any(), "Log directory names are not unique."
    args_df.index = args_df.index.rename('Run')

    return args_df


def get_learning_data(log_dirs, file_reader=load_results, verbose=1):
    """Load learning curve data from selected log directories
    and return as a list of pandas dataframes.
    """

    learning_curves = {}

    learning_filename = log_filenames['learning']
    if verbose > 1:
        print(f"\nLoading data from '{learning_filename}' files...\n")

    for dir_path in log_dirs:
        dir_name = os.path.split(dir_path)[-1]
        try:
            learning_data = load_results(dir_path)
        except FileNotFoundError:
            if verbose > 0:
                print(f"{dir_name}: WARNING. No '{learning_filename}' "
                      f"file in: {dir_path}")
        except:
            if verbose > 0:
                print(f"{dir_name}: WARNING. Could not read '{learning_filename}' "
                      f"file in: {dir_path}")
        else:
            if verbose > 1:
                print(f"{dir_name}: {len(learning_data)} data points")
            if dir_name in learning_curves:
                raise ValueError(f"Log directory {dir_name} is not unique.")
            learning_curves[dir_name] = learning_data

    if verbose > 0:
        print(f"{len(learning_curves)} '{learning_filename}' "
              f"files found")

    return learning_curves

EPISODES_WINDOW = 100
plot_colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow',
               'black', 'purple', 'pink', 'brown', 'orange', 'teal',
               'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
               'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple',
               'darkred', 'darkblue']


def convert_learning_data(timesteps, xaxis='timesteps'):
    """
    Converts x-axis values to appropriate units

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) units for the axis ('timesteps',
        'episodes' or 'walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """
    if xaxis == 'timesteps':
        x_values = np.cumsum(timesteps.l.values)
        y_values = timesteps.r.values
    elif xaxis == 'episodes':
        x_values = np.arange(len(timesteps))
        y_values = timesteps.r.values
    elif xaxis == 'walltime_hrs':
        x_values = timesteps.t.values / 3600.
        y_values = timesteps.r.values
    else:
        raise NotImplementedError
    return pd.Series(y_values, index=x_values)


def plot_learning_curves(combined_data, xlabel=None, ylabel=None,
                         ylim=None, values='Min|Mean|Max', label=None,
                         rolling_mean=True, window=100,
                         color='b', title="Learning Curves"):

    values_to_plot = values.strip().split('|')
    assert len(values_to_plot) == 3, "Invalid values argument."
    data = []
    for v in values_to_plot:
        if v == 'Min':
            data.append(combined_data.min(axis=1).rename(v))
        elif v == 'Max':
            data.append(combined_data.max(axis=1).rename(v))
        elif v == 'Mean':
            data.append(combined_data.mean(axis=1).rename(v))
        elif v == 'Median':
            data.append(combined_data.median(axis=1).rename(v))
        elif 0 <= int(v)< 100:
            data.append(combined_data.quantile(int(v)/100).rename(f"{v}%"))
        else:
            raise ValueError("Values argument not recognized.")

    if rolling_mean:
        data[1] = data[1].rolling(window=window).mean()
        data[1] = data[1].rename(f"Rolling {data[1].name}")

    if xlabel is None:
        xlabel = combined_data.index.name

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()

    if label is None:
        label = data[1].name

    plt.fill_between(combined_data.index, data[0], data[2],
                     alpha=0.1, color=color)
    plt.plot(combined_data.index, data[1], '-',
             color=color, label=label)

    plt.legend(loc="best")

