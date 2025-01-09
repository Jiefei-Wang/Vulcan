from tqdm import tqdm
import pandas as pd


def get_text(row):
    import pandas as pd
    if type(row) == pd.Series:
        return row['concept_name']
    else:
        return row['concept_name'].values[0]


conceptML1 = pd.read_feather('data/ML/conceptML1.feather')
concept_relationship = pd.read_feather('data/omop_feather/concept_relationship.feather')

## get non-standard conditions
std_condition = conceptML1[conceptML1['domain_id'] == 'Condition']
std_condition.groupby("vocabulary_id").size()

std_condition2 = std_condition[std_condition.apply(lambda x: len(x['text']) != 0, axis=1)].copy()
std_condition2['std_name'] = std_condition2['domain_id'] + ': ' + std_condition2['concept_name']


std_condition3 = std_condition2.copy()


n_neg = 4
dataset = []
for i in tqdm(range(len(std_condition2))):
    row = std_condition2.iloc[i]
    concept_name = row['concept_name']
    std_name = row['std_name']
    ## positive sample
    text = row['text'].tolist() + [concept_name]
    pos_dt = [{'sentence1': std_name, 'sentence2': i, 'label': "equals"} for i in text]
    
    ## negative sample
    





def negative_sample(df, n_neg, excludes):
    descriptions = []
    while len(descriptions) < n_neg:
        idx = np.random.randint(0, len(df))
        desc = df.iloc[idx]['description']
        if type(desc) != str:
            raise ValueError(f"{idx}: Description is not a string: {desc}")
        if desc not in descriptions and desc not in excludes:
            descriptions.append(desc)
    return descriptions
    
def get_sample(row, df, n_neg):
    row_description = get_description(row)
    std_description = [get_description(df[df['concept_id'] == pos]) for pos in row['std_concept_id']]
    dataset.append({'premise': row_description, 'hypothesis': std_description, 'label': 1})
    
    ## sample negative samples, It can possibly contain the standard description or itself
    ## so we sample n_neg + len(std_description) samples
    neg_descriptions = negative_sample(df, n_neg, std_description+[row_description])
    dataset = []
    for j in range(n_neg):
        dataset.append({'premise': row_description, 'hypothesis': neg_descriptions[j], 'label': 0})
    return dataset




## for each non-standard condition, find the name of the standard condition, and n negative samples
## The dataset column: premise, hypothesis, label(0 or 1)
import numpy as np
n_neg = 4
dataset = []
for i in tqdm(range(len(nonstd_condition))):
    row = nonstd_condition.iloc[i]
    row_description = row['description']
    std_description = [conceptEX.loc[pos]['description'] for pos in row['std_concept_id']]
    for j in range(len(std_description)):
        dataset.append({'premise': row_description, 'hypothesis': std_description[j], 'label': 1})
    
    ## sample negative samples, It can possibly contain the standard description or itself
    ## so we sample n_neg + len(std_description) samples
    neg_descriptions = negative_sample(conceptEX, n_neg, std_description+[row_description])
    for j in range(n_neg):
        dataset.append({'premise': row_description, 'hypothesis': neg_descriptions[j], 'label': 0})


# conceptEX.iloc[285757]

# ## to huggingface dataset
# dataset2 = pd.DataFrame(dataset)
# dataset2['hypothesis']
# ## find which values in hypothesis column contains non-string object
# dataset2[~dataset2['premise'].apply(lambda x: isinstance(x, str))]


from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from datasets import Dataset

dataset2 = Dataset.from_list(dataset)
## names to sentence1, sentence2
dataset2 = dataset2.rename_column('premise', 'sentence1')
dataset2 = dataset2.rename_column('hypothesis', 'sentence2')
## save train_dataset to disk
dataset2.save_to_disk('data/train_dataset')
