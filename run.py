import os
from modules.CodeBlockExecutor import execute_and_embed

with open('reload_library.py') as f:
    exec(f.read())

########################
## Basic data conversion
## Only need to run once
########################
with open('scripts/base_data/1_data_conversion.py') as f:
    exec(f.read())


########################
## matching data
## Only need to run once
########################
with open('scripts/matching/1_extract.py') as f:
    exec(f.read())

execute_and_embed('scripts/matching/2_combined.py')


execute_and_embed('scripts/matching/3_train_valid_split.py')


# Unit test
with open('scripts/matching/test_dataset.py', encoding="UTF-8") as f:
    exec(f.read())



########################
## Prepare ML data
## Only need to run once
########################

with open('scripts/ML/1_init_false_positive.py') as f:
    exec(f.read())


with open('scripts/ML/2_train.py') as f:
    exec(f.read())




########################
## Train
########################
with open('scripts/ML_train.py') as f:
    exec(f.read())


########################
## sanity check
########################
with open('sanity/sanity_ML_data.py') as f:
    exec(f.read())

## TODO: add sanity check for ML_data_condition_matching


from modules.CodeBlockExecutor import execute_and_embed
execute_and_embed('scripts/test2.py')
