with open('reload_library.py') as f:
    exec(f.read())

########################
## Convert OMOP and UMLS data to feather format
## Only need to run this once
########################
with open('scripts/data_conversion.py') as f:
    exec(f.read())

########################
## Data management
########################
## TODO: optimize the performance
with open('scripts/ML_data.py') as f:
    exec(f.read())


with open('scripts/ML_data_condition_target.py') as f:
    exec(f.read())

with open('scripts/ML_data_condition_matching.py') as f:
    exec(f.read())

with open('scripts/ML_data_condition_relation.py') as f:
    exec(f.read())

with open('scripts/ML_FP_condition_matching.py') as f:
    exec(f.read())

with open('scripts/ML_train2.py') as f:
    exec(f.read())


########################
## sanity check
########################
with open('sanity/sanity_ML_data.py') as f:
    exec(f.read())

## TODO: add sanity check for ML_data_condition_matching