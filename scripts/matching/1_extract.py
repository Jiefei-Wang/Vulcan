import os
from modules.CodeBlockExecutor import execute_and_embed

if not os.path.exists('data/matching'):
    os.makedirs('data/matching')

execute_and_embed('scripts/matching/data/1_UMLS.py')
execute_and_embed('scripts/matching/data/2_omop.py')
