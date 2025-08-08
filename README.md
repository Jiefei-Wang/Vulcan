# Agreement

- sentence1 is always the standard concept
- sentence2 is the non-standard for training


# UMLS
- How many concepts get mapped to OMOP? By vocabulary?
- fix those vocabularies codes: {'HGNC', 'MDR', 'MTHSPL', 'MED-RT', 'CDT'} that does not have a mapping to OMOP

# scripts/matching/3_train_valid_split.py

The number of concepts in the condition domain:
```python
std_condition_concept['concept_id'].nunique() # 160288
condition_matching_map_train['concept_id'].nunique()  # 104672

condition_matching_map_train.groupby(['source', 'type'])['concept_id'].nunique()
# source  type   
# OMOP    nonstd     83257
#         synonym    98677
# UMLS    DEF        19553
#         STR        49962
```
It cannot cover all concepts, so we need more data.

# vocabulary table
CDISC:
| vocabulary | concept_code |  type | name |
|------------|--------------|-------|------|
|   CDISC    |    C62266    |  name  |  ACCELERATED...|
|   CDISC    |    C62266    |  def  |  An electrocardiographic findin...|
|   CDISC    |    C62266    |  synonym  | Accelerated idioventricular rhythm |


# map_table specification

| concept_id | source | source_id | type | name |
|------------|--------|-----------|------|------|
|   1        |  umls  |   XX      |  STR |  YY  |

- concept_id: The standard OMOP concept ID
- source: The source vocabulary of the concept (e.g., 'umls', 'cdisc')
- source_id: The unique identifier of the concept in the source vocabulary
- type: The type of the name in the source vocabulary
- name: The name of the concept in the source vocabulary 


# concept mapping design

name_table: each row is a unique name from a particular source

| name_id | source | source_id | type | name |
|---------|--------|-----------|------|------|
|   1     |  umls  |   XX      |  STR |  YY  |


name_bridge: each row is a mapping from a concept id to a name_id

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

## Install FAISS-GPU
```
conda install conda-forge::faiss-gpu
```



# Idea
unknown: prostate cancer
std: Malignant tumor of prostate

parent: Malignant neoplasm of abdomen


- find top 100 similar std concepts to: prostate cancer
    - top 1: Malignant tumor of prostate, score: 0.5
- find all parents of Malignant tumor of prostate
    - parents = ['Malignant neoplasm of abdomen', 'Malignant tumor of pelvis', ...]

- calculate similarity between 
    - 'Malignant neoplasm of abdomen' and  "<|parent of|> prostate cancer", score: 0.7 
    - 'Malignant tumor of pelvis' and  "<|parent of|> prostate cancer" , score: 0.6 
- Combine direct matching and relation scores 0.5 + 0.6