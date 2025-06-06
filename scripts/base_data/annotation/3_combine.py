# Combine all mapping into one file
# File: data/base_data/concept_names.feather

import os
from tqdm import tqdm
import pandas as pd
import numpy as np
# import swifter
from modules.timed_logger import logger
import duckdb

# Load OMOP CDM concept tables from feather files
concept = pd.read_feather('data/omop_feather/concept.feather')
std_bridge = pd.read_feather("data/omop_feather/std_bridge.feather")
nonstd_names = pd.read_feather('data/base_data/nonstd_names.feather')
synonum_names = pd.read_feather('data/base_data/synonum_names.feather')
umls_names = pd.read_feather('data/base_data/umls_names.feather')


logger.log("Merge all concept information")
all_names = pd.concat([nonstd_names, synonum_names, umls_names], ignore_index=True)[['concept_id', 'source_name', 'source_id', 'source', 'type']]
# [18803072 rows x 5 columns]

all_names.columns
# ['concept_id', 'source_name', 'source_id', 'source', 'type']

all_names.groupby(['source', 'type'])['source_name'].count()
# source  type
# OMOP    nonstd     9683303
#         synonym    4134191
# UMLS    DEF         638661
#         STR        4346917

# to lower and strip
all_names['source_name'] = all_names['source_name'].str.lower().str.strip()
# TODO: Inspect all_names for potential issues in source_name 

all_names = all_names.drop_duplicates(['concept_id', 'source_name'])
# [15192399 rows x 5 columns]

concept_names = all_names.merge(
    std_bridge,
    left_on='concept_id',
    right_on='concept_id'
).drop(columns=['concept_id']).rename(
    columns={
        'std_concept_id': 'concept_id'
    })
# [11467354 rows x 5 columns]

concept_names['source_id'] = concept_names['source_id'].astype(str)

logger.log("Saving concept_names")

path = 'data/base_data'
## create directory if it does not exist
if not os.path.exists(path):
    os.makedirs(path)

concept_names.to_feather(os.path.join(path, 'concept_names.feather'))

logger.done()

