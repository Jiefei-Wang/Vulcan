import importlib
import modules.ML_sampling
importlib.reload(modules.ML_sampling)
import modules.ML_data
importlib.reload(modules.ML_data)
import modules.ChromaVecDB
importlib.reload(modules.ChromaVecDB)
import modules.CodeBlockExecutor
importlib.reload(modules.CodeBlockExecutor)
import modules.FalsePositives
importlib.reload(modules.FalsePositives)


import modules.Dataset
importlib.reload(modules.Dataset)

import modules.FaissDB
importlib.reload(modules.FaissDB)

# def load_lib(libPath):
#     with open(libPath) as f:
#         exec(f.read(), globals())

# load_lib('modules/ML_sampling.py')
# load_lib('modules/ML_data.py')
# load_lib('modules/ChromaVecDB.py')

