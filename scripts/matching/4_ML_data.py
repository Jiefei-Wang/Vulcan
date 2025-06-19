import pandas as pd
import os
from modules.timed_logger import logger
from modules.Iterable import PositiveIterable, NegativeIterable, FalsePositiveIterable, CombinedIterable
from tqdm import tqdm
logger.reset_timer()
logger.log("")

base_path = "data/matching"
std_condition_concept = pd.read_feather(os.path.join(base_path, 'std_condition_concept.feather'))
condition_matching_name_bridge_train = pd.read_feather(os.path.join(base_path, 'condition_matching_name_bridge_train.feather'))
condition_matching_name_table_train = pd.read_feather(os.path.join(base_path, 'condition_matching_name_table_train.feather'))


with open('modules/Iterable.py') as f:
    exec(f.read())


pos_it = PositiveIterable(
    target_concepts=std_condition_concept,
    name_table=condition_matching_name_table_train,
    name_bridge=condition_matching_name_bridge_train,
    max_element=10
)

dt = []
for i, item in tqdm(enumerate(pos_it), total=len(std_condition_concept)*10):
    dt.append(item)
    pass

df = pd.DataFrame(dt)
df

## count by concept_id1
df.groupby(['concept_id1']).size().reset_index(name='count').sort_values(by='count', ascending=False)
#         concept_id1  count
# 0             22274     10
# 81981      37163123     10
# 46764       4162134     10
# 46766       4162137     10
# 46771       4162218     10
# 101280     44811661      1
# 101281     44811662      1
# 101282     44811663      1
# 101283     44811664      1
# 104708     46287268      1


neg_it = NegativeIterable(
    target_concepts=std_condition_concept,
    name_table=condition_matching_name_table_train,
    blacklist_name_bridge=condition_matching_name_bridge_train,
    max_element=10, 
    seed=42
)

for i, item in tqdm(enumerate(neg_it), total=len(std_condition_concept)*10):
    # print(f"{i+1}: {item}")
    # if i >= 1000000:
    #     break
    pass


com_it = CombinedIterable(
    target_concepts=std_condition_concept,
    name_table=condition_matching_name_table_train,
    positive_name_bridge=condition_matching_name_bridge_train,
    blacklist_name_bridge=condition_matching_name_bridge_train,
    false_positive_name_bridge=condition_matching_name_bridge_train,
    positive_max_element=10,
    false_positive_max_element=10,
    negative_max_element=10
)


dt = []
for i, item in tqdm(enumerate(com_it), total=len(std_condition_concept)*30):
    dt.append(item)
    # print(f"{i+1}: {item}")
    # if i >= 1000000:
    #     break

df = pd.DataFrame(dt)
df

df[['concept_id1', 'concept_id2', 'label', 'iter_id', 'rank']]

df.iloc[(1000000-30):(1000000-20)][['concept_id1', 'concept_id2', 'label', 'iter_id', 'rank']]

df.groupby(['iter_id']).size().reset_index(name='count')
#           iter_id    count
# 0  false_positive   654497
# 1        negative  1602880
# 2        positive   654497

a=condition_matching_name_bridge_train.merge(
    std_condition_concept[['concept_id', 'concept_name']],
    on='concept_id',
    how='inner'
).merge(
    condition_matching_name_table_train,
    on='name_id',
    how='inner'
)

## find the number of maps for each concept_id
counts = a.groupby('concept_id').size().reset_index(name='count').sort_values(by='count', ascending=False)

## max out at 10
counts['count'] = counts['count'].clip(upper=10)

#         concept_id  count
# 13131       761978     10
# 30136      4080451     10
# 60406      4265194     10
# 55782      4224767     10
# 2154        195584     10
# 101280    44811661      1
# 101281    44811662      1
# 101282    44811663      1
# 101283    44811664      1
# 104708    46287268      1

#sum 
counts['count'].sum()  # 654481


40305987

condition_matching_name_bridge_train[condition_matching_name_bridge_train.concept_id == 40305987]
concept[concept.concept_id == 40305987]

a[a.concept_id == 40305987]