- omop_data.py: Get OMOP data from the database
- ML_data.py: Prepare data for NLP
- ML_train.py: Train the model
- test.py: Test the model


# UMLS
- How many concepts get mapped to OMOP? By vocabulary?
- fix those vocabularies codes: {'HGNC', 'MDR', 'MTHSPL', 'MED-RT', 'CDT'} that does not have a mapping to OMOP

# scripts/base_data/annotation/3_combine.py
- Inspect all_names for potential issues in source_name


# vocabulary table
CDISC:
| vocabulary | concept_code |  type | name |
|------------|--------------|-------|------|
|   CDISC    |    C62266    |  name  |  ACCELERATED...|
|   CDISC    |    C62266    |  def  |  An electrocardiographic findin...|
|   CDISC    |    C62266    |  synonym  | Accelerated idioventricular rhythm |


# concept mapping design

name_table: each row is a unique name from a particular source

| name_id | source | source_id | type | name |
|---------|--------|-----------|------|------|
|   1     |  umls  |   XX      |  STR |  YY  |


concept_name_map: each row is a mapping from a concept id to a name_id

| concept_id | name_id |
|------------|---------|
|   123      |  1      |


```python
std_conditions.groupby('vocabulary_id')['concept_id'].count()
"""
HCPCS                    1
ICDO3                56858
Nebraska Lexicon      1274
OMOP Extension         341
OPCS4                    1
SNOMED               98720
SNOMED Veterinary     3093
"""

nonstd_conditions = nonstd_concept[nonstd_concept.domain_id == 'Condition']
nonstd_conditions.groupby('vocabulary_id')['concept_id'].count()
"""
CDISC                   455
CIEL                  38818
CIM10                 13885
CO-CONNECT               16
Cohort                   66
HemOnc                  260
ICD10                 14113
ICD10CM               88510
ICD10CN               30588
ICD10GM               15952
ICD9CM                14929
ICDO3                  5677
KCD7                  19705
MeSH                  12343
Nebraska Lexicon     150062
OMOP Extension            8
OPCS4                     5
OXMIS                  5704
OncoTree                885
PPI                      74
Read                  47836
SMQ                     324
SNOMED                58172
SNOMED Veterinary       144
"""
```