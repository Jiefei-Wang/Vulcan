
def find_indices(lst, values):
    return [find_index(lst, value) for value in values]

def find_index(lst, value):
    try:
        return lst.index(value)
    except ValueError:
        return None

def map_concepts(db, df, n_results=100):
    results = db.query(
        df,
        n_results = n_results
    )

    ## id to int
    maps_to = [[int(i) for i in x] for x in results['ids']]
    df2 = df.copy()
    df2['maps_to'] = maps_to

    ## look over results and see 
    ## the location of the standard concept id in the list
    correct_index = []
    for i in range(len(df2)):
        row = df2.iloc[i]
        idx = find_indices(row.maps_to, row.std_concept_id)
        # std_num = len(row.std_concept_id)
        # scores = [k - std_num + 1 if k!=None else None for k in idx]
        correct_index.append(idx)

    df2['correct_index'] = correct_index
    return df2


def calculate_score(indice):
    if None in indice:
        return 99999
    else:
        std_num = len(indice) # 3
        scores = max([k - std_num + 2 for k in indice])
        return scores

def performance_metrics(df, k=50):
    ## check if the correct index is in the top k
    df['topk'] = df.correct_index.apply(lambda x: calculate_score(x) <= k)

    ## accuracy
    return df['topk'].mean()
