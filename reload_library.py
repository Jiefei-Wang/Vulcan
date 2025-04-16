import importlib
import modules.ML_sampling
import modules.ML_data
import modules.ChromaVecDB
importlib.reload(modules.ML_sampling)
importlib.reload(modules.ML_data)
importlib.reload(modules.ChromaVecDB)


def load_lib(libPath):
    with open(libPath) as f:
        exec(f.read(), globals())

load_lib('modules/ML_sampling.py')
load_lib('modules/ML_data.py')
load_lib('modules/ChromaVecDB.py')

