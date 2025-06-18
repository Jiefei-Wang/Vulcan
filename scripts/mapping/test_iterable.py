from modules.Iterable import PositiveIterable, NegativeIterable, FalsePositiveIterable, CombinedIterable
import pandas as pd

with open('modules/Iterable.py') as f:
    exec(f.read())

target_concepts = pd.DataFrame({
    'concept_id': [101, 102, 103],
    'concept_name': ['Diabetes', 'Hypertension', 'Asthma']
})

name_table = pd.DataFrame({
    'name_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'source': ['ICD10', 'ICD10', 'SNOMED', 'SNOMED', 'ICD9', 'ICD9', 'LOINC', 'LOINC', 'CPT', 'CPT'],
    'source_id': ['E11', 'I10', '73211009', '195967001', '250', '401', '33747-0', '8480-6', '99213', '99214'],
    'type': ['diagnosis', 'diagnosis', 'diagnosis', 'diagnosis', 'diagnosis', 'diagnosis', 'lab', 'vital', 'procedure', 'procedure'],
    'name': ['Diabetes1', 'Hypertension1', 'Diabetes2', 'Asthma1', 'Diabetes3', 'Hypertension2', 'Glucose', 'Blood pressure', 'Office visit', 'Asthma2']
})

name_bridge = pd.DataFrame({
    'concept_id': [101, 101, 101, 102, 102, 103, 103],
    'name_id': [1, 3, 5 ,2, 6, 4, 10]  
})


# Test positive iterable
print("Testing PositiveIterable:")
pos_it = PositiveIterable(
    target_concepts=target_concepts,
    name_table=name_table,
    name_bridge=name_bridge,
    max_element=5
)

for i, item in enumerate(pos_it):
    print(f"Positive {i+1}: {item}")




# negative_name_bridge = pd.DataFrame({
#     'concept_id': [101, 101, 103],
#     'name_id': [2, 4, 9]  # Example negative names
# })

neg_it = NegativeIterable(
    target_concepts=target_concepts,
    name_table=name_table,
    blacklist_name_bridge=name_bridge, 
    max_element=1,  # Reduced for faster testing
    seed=42
)

for i, item in enumerate(neg_it):
    print(f"Negative {i+1}: {item}")


false_positive_name_bridge = pd.DataFrame({
    'concept_id': [101, 101, 103],
    'name_id': [2, 4, 9]  # Example false positive names
})

fp_it = FalsePositiveIterable(
    target_concepts=target_concepts,
    name_table=name_table,
    name_bridge=false_positive_name_bridge,
    max_element=3  # Reduced for faster testing
)


for i, item in enumerate(fp_it):
    print(f"{i+1}: {item}")




com_it = CombinedIterable(
    target_concepts=target_concepts,
    name_table=name_table,
    positive_name_bridge=name_bridge,
    blacklist_name_bridge = name_bridge,
    false_positive_name_bridge=false_positive_name_bridge,
)

for i, item in enumerate(com_it):
    print(f"{i+1}: {item}")

