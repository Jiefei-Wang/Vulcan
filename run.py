
########################
## Convert OMOP and UMLS data to feather format
########################
with open('scripts/data_conversion.py') as f:
    exec(f.read())

########################
## Data management
########################
with open('reload_library.py') as f:
    exec(f.read())


with open('scripts/ML_data.py') as f:
    exec(f.read())
    
with open('scripts/ML_data_condition_matching.py') as f:
    exec(f.read())
    
    
########################
## sanity check
########################
with open('sanity/sanity_ML_data.py') as f:
    exec(f.read())