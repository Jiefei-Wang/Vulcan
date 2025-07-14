# This script generate false postives samples for training data


## For Windows: Python 3.10, chromaDB version 0.5.4
## For Windows: Python 3.11, chromadb==0.5.0 chroma-hnswlib==0.7.3
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Fix for ONNX Runtime DLL issues on Windows
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['ONNXRUNTIME_PROVIDERS'] = 'CPUExecutionProvider'

## without this, conda might give an error when loading chromadb
import onnxruntime
import pandas as pd
import duckdb
from modules.ChromaVecDB import ChromaVecDB
from modules.ModelFunctions import auto_load_model
from modules.timed_logger import logger


# n_fp_matching = 50
def get_false_positives(model, target_concepts, n_fp_matching = 50):
    logger.reset_timer()
    ####################################
    ## Work on conditions domain
    ####################################
    logger.log("Building reference embedding for standard concepts")
    db = ChromaVecDB(model=model, name="ref")
    db.empty_collection()
    db.store_concepts(target_concepts, batch_size= 5461)


    logger.log("Querying reference embedding")
    ## for each item in candidate_df_matching
    n_results = n_fp_matching * 2 + 10
    results = db.query(
            target_concepts,
            n_results = n_results
        )

    logger.log("Building candidate false positive dataframe")
    candidate_fp = target_concepts[['concept_id', "concept_name"]].copy()
    candidate_fp['maps_to'] = [[int(i) for i in x] for x in results['ids']]
    candidate_fp['distance'] = results['distances']
    candidate_fp = candidate_fp.explode(['maps_to', 'distance'], ignore_index=False)
    
    
    ## For each concept_id, sort by distance and keep the first n_fp_matching entries
    candidate_fp = duckdb.query(f"""
        -- Rank candidates by distance for each concept_id
        WITH ranked_candidates AS (
        SELECT *,
                ROW_NUMBER() OVER (PARTITION BY concept_id ORDER BY distance) AS rn
        FROM candidate_fp
        ),
        -- Select the top n_fp_matching candidates for each concept_id
        -- while removing 0 distance candidates
        top_candidates AS (
            SELECT concept_id as concept_id1, 
            concept_name as sentence1, 
            maps_to as concept_id2, 
            distance
            FROM ranked_candidates
            WHERE rn <= {n_fp_matching} and distance > 0
            ORDER BY concept_id, distance
        )
        SELECT concept_id1, sentence1, 
        concept_id2, target_concepts.concept_name as sentence2, 
        distance
        FROM top_candidates
        INNER JOIN target_concepts
        ON top_candidates.concept_id2 = target_concepts.concept_id;
        """).df()


    logger.done()
    
    return candidate_fp



