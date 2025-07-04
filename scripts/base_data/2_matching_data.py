# Create a mapping from standard to non-standard concepts
# Table: concept_names
# columns: ['concept_id', 'source_name', 'source_id', 'source', 'type']
# 
# Store the result in data/base_data/concept_names.feather


import os
from modules.timed_logger import logger
logger.reset_timer()


path = 'data/base_data'
## create directory if it does not exist
if not os.path.exists(path):
    os.makedirs(path)


with open('scripts/base_data/annotation/1_UMLS.py') as f:
    exec(f.read())


with open('scripts/base_data/annotation/2_omop.py') as f:
    exec(f.read())


with open('scripts/base_data/annotation/3_combine.py') as f:
    exec(f.read())


