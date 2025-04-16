import pandas as pd
import os
from multiprocessing import Pool

def read_csv_chunk(filename):
    """Reads a CSV chunk into a DataFrame."""
    return pd.read_csv(filename)

def read_csv_parallel(file_list, num_processes=os.cpu_count()):
    """Reads multiple CSV files in parallel and concatenates them."""
    with Pool(processes=num_processes) as pool:
        df_list = pool.map(read_csv_chunk, file_list)
    return pd.concat(df_list, ignore_index=True)
