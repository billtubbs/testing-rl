import os
import datetime
import csv


def make_new_filename(name, dir_path='./', start_time=None, ext='csv'):
    """Checks the existing files at filepath and makes a
    new (unique) filename from name in the following format:
    
    '{name}-%Y-%m-%d-{counter}.{ext}'

    Where n is an integer that increments by 1 until the
    filename is unique.

    Example, if there is not file in the current directory:
    >>> make_new_filename('testing', ext='csv')                                                                                      
    'testing-2019-10-28.csv'

    Example, if the file already exists:
    >>> make_new_filename('testing', ext='csv')                                                                                      
    'testing-2019-10-28-1.csv'
    """

    if start_time is None:
        start_time = datetime.datetime.now()
    date_string = start_time.strftime('%Y-%m-%d')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = f'{name}-{date_string}.{ext}'
    existing_files = os.listdir(dir_path)
    counter = 0
    while filename in existing_files:
        counter += 1
        filename = f'{name}-{date_string}-{counter}.{ext}'
    return filename


def write_to_csv_file(filepath, count, values, reward, mode='a',
                      delimiter=','):
    with open(filepath, mode=mode) as f:
        csv_writer = csv.writer(f, delimiter=delimiter)
        csv_writer.writerow([count] + list(values) + [reward])
