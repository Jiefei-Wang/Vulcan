import os

if not os.path.exists('data/matching'):
    os.makedirs('data/matching')

with open('scripts/matching/data/1_UMLS.py') as f:
    exec(f.read())

with open('scripts/matching/data/2_omop.py') as f:
    exec(f.read())

