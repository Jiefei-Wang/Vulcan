from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)


# # Load model from HuggingFace Hub
# tokenizer = AutoTokenizer.from_pretrained('./all-MiniLM-L6-v2')
# model = AutoModel.from_pretrained('./all-MiniLM-L6-v2')

# #Mean Pooling - Take attention mask into account for correct averaging
# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output[0] #First element of model_output contains all token embeddings
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# # Sentences we want sentence embeddings for
# sentences = ['This is an example sentence', 'Each sentence is']

# # Tokenize sentences
# encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# # Compute token embeddings
# with torch.no_grad():
#     model_output = model(**encoded_input)

# model_output.last_hidden_state.shape



# # Perform pooling
# sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# # Normalize embeddings
# sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

# print("Sentence embeddings:")
# print(sentence_embeddings)





def get_text(row):
    import pandas as pd
    if type(row) == pd.Series:
        return row['concept_name']
    else:
        return row['concept_name'].values[0]

conceptEX = pd.read_feather('data/omop_feather/conceptEX.feather')
concept_relationship = pd.read_feather('data/omop_feather/concept_relationship.feather')



conceptEX.columns

conceptEX.vocabulary_id.unique()


# conceptEX['text'] = conceptEX.parallel_apply(text, axis=1)

## get non-standard conditions
condition = conceptEX[conceptEX['domain_id'] == 'Condition'].copy()


condition['text'] = condition['domain_id'] + ':' + condition['concept_name']
condition['text_name_only'] = condition['concept_name']




std_condition = condition[condition['standard_concept'] == 'S']
nonstd_condition = condition[condition['standard_concept'] != 'S']



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
