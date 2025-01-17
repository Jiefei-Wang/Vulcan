# This file create training data for the ML model
# 
# Depends: data/ML/conceptML.feather
# 
# Sampling strategy:
# - For each non-standard condition, create positive samples
# - For each non-standard condition, sample n_neg(=4) negative samples
#
# Output: pd.DataFrame, 
# columns=['sentence1', 'sentence2', 'concept_id1', 'concept_id2', 'label1']
# label1 = 1 if sentence1 maps to sentence2, 0 otherwise
# label2 = 1 if sentence1 is a parent of sentence2, 0 otherwise
# label3 = 1 if sentence1 is a child of sentence2, 0 otherwise


# Bert -> Embeddings -> FC(n input, n output) -> Cosine Similarity <- whether two sentences are similar
# positive: high similarity
# negative: low similarity
# loss1 = negative - positive
# 
# Bert -> Embeddings -> FC(n input, n output) -> Cosine Similarity <- whether two sentences have parent-child relationship
# positive: high similarity
# negative: low similarity
# loss2 = negative - positive
#
# total loss = loss1 + loss2 + loss3