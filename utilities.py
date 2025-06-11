import os
import csv
import time
import numpy as np

def log_coordinator_timing(pso_time, CSV_FILE):
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['centralised', 'pso_time'])
        writer.writerow(["yes", pso_time])




def write_np_to_file(array, filename:str, folder) -> None:
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(1, -1)
    np.savetxt(filepath, array, fmt='%d', delimiter=' ')

def write_to_file(input , filename: str) -> None:
    with open('data/' + filename + '.txt', 'w') as f:
        f.write(str(input))