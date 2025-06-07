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
CDOSC:
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