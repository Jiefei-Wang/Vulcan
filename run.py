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
from modules.CodeBlockExecutor import execute_and_embed

with open('scripts/matching/1_extract.py') as f:
    exec(f.read())

execute_and_embed('scripts/matching/2_combined.py')

execute_and_embed('scripts/matching/3_condition_domain.py')

execute_and_embed('scripts/matching/4_train_test_valid.py')

execute_and_embed('scripts/matching/4_false_positive.py')


########################
## relation data
## Only need to run once
########################

from modules.CodeBlockExecutor import execute_and_embed

execute_and_embed('scripts/relation/1_positive_relation.py')


execute_and_embed('scripts/relation/2_false_positive.py')


########################
## Prepare ML data
## Only need to run once
########################
from modules.CodeBlockExecutor import execute_and_embed


with open('scripts/ML/1_train.py', encoding="UTF-8") as f:
    exec(f.read())


with open('scripts/ML/2_test.py', encoding="UTF-8") as f:
    exec(f.read())

########################
## sanity check
########################
with open('sanity/sanity_ML_data.py') as f:
    exec(f.read())




exec(open('tests/test_dataset.py').read())

exec(open('tests\test_FaissDB.py').read())


exec(open('tests/test_FalsePositives.py').read())
