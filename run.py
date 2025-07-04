import os

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
# create data/matching
import os
if not os.path.exists('data/matching'):
    os.makedirs('data/matching')

with open('scripts/matching/1_UMLS.py') as f:
    exec(f.read())

with open('scripts/matching/2_omop.py') as f:
    exec(f.read())

with open('scripts/matching/3_train_valid_split.py') as f:
    exec(f.read())

# Unit test
with open('scripts/matching/test_dataset.py') as f:
    exec(f.read())



########################
## Prepare ML data
## Only need to run once
########################
with open('scripts/ML_data/condition_target.py') as f:
    exec(f.read())

with open('scripts/ML_data/condition_matching.py') as f:
    exec(f.read())

with open('scripts/ML_data/condition_relation.py') as f:
    exec(f.read())

with open('scripts/ML_FP_condition_matching.py') as f:
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