# 1. remove all non-english rows
# 2. remove duplicated names identified by name_stripped
# 3. remove mapping that maps to the concept name itself

import pandas as pd
import os
from modules.timed_logger import logger
import duckdb
from modules.CodeBlockExecutor import trace, tracedf

logger.reset_timer()
logger.log("Combining all map_tables")

std_bridge = pd.read_feather("data/omop_feather/std_bridge.feather")
concept= pd.read_feather('data/omop_feather/concept.feather')


input_dir = "data/matching"
output_dir = "data/matching"
pairs_list = [
    'OMOP_pairs.feather',
]

####################
## Combine all map_tables
####################
combined_pairs = pd.concat(
    [pd.read_feather(os.path.join(input_dir, name_map_table)) for name_map_table in pairs_list],
    ignore_index=True
)
combined_pairs['source_id'] = combined_pairs['source_id'].astype(str)

tracedf(combined_pairs)
#> DataFrame dimensions: 6384688 rows × 5 columns
#> Column names:
#> ['concept_id', 'source', 'source_id', 'type', 'name']
#> Estimated memory usage: 1.87 GB

# - For each concept_id in combined_pairs, map to a standard concept_id using std_bridge
#   1. if the concept_id is standard, it will be mapped to itself in the std_bridge
#   2. if the concept_id is non-standard, it will be mapped to a standard concept_id in the std_bridge
#   3. For those that are not in the std_bridge, there is no way to map them 
# to a standard concept, so we will not use them in the training.
# - For name_stripped, keep only letters in name, all lowercase from name
# - Remove empty names
matching_pairs = duckdb.query("""
    SELECT std_bridge.std_concept_id AS concept_id, source, source_id, type, name,
    LOWER(REGEXP_REPLACE(name, '[^a-zA-Z0-9]', '', 'g')) AS name_stripped
    FROM combined_pairs
    inner join std_bridge
    ON combined_pairs.concept_id = std_bridge.concept_id
    where name IS NOT NULL AND name != ''
""").df()

tracedf(matching_pairs)
#> DataFrame dimensions: 5491232 rows × 6 columns
#> Column names:
#> ['concept_id', 'source', 'source_id', 'type', 'name', 'name_stripped']
#> Estimated memory usage: 2.13 GB


## remove non-english rows like: 인도신1mg주
matching_pairs = matching_pairs[matching_pairs['name'].apply(lambda x: x.isascii())].reset_index(drop=True)

tracedf(matching_pairs)
#> DataFrame dimensions: 4835632 rows × 6 columns
#> Column names:
#> ['concept_id', 'source', 'source_id', 'type', 'name', 'name_stripped']
#> Estimated memory usage: 1.88 GB

## for each concept_id, remove the duplicates in name_stripped
matching_pairs = duckdb.query("""
    SELECT *
    FROM (
        SELECT *, 
            ROW_NUMBER() OVER (PARTITION BY concept_id, name_stripped ORDER BY source_id) AS rn
        FROM matching_pairs
    ) t
    WHERE rn = 1;
"""
).df()

tracedf(matching_pairs)
#> DataFrame dimensions: 4437904 rows × 7 columns
#> Column names:
#> ['concept_id', 'source', 'source_id', 'type', 'name', 'name_stripped', 'rn']
#> Estimated memory usage: 1.77 GB

# remove the mapping that maps to the concept name itself
concept_id_to_name = concept[['concept_id', 'concept_name']]
matching_pairs = duckdb.query("""
    with concept_id_to_name2 as (
        select concept_id, concept_name, LOWER(REGEXP_REPLACE(concept_name, '[^a-zA-Z0-9]', '', 'g')) AS concept_name_stripped
        from concept_id_to_name
    )
    SELECT concept_id_to_name2.concept_id, concept_id_to_name2.concept_name, concept_id_to_name2.concept_name_stripped, source, source_id, type, name, name_stripped
    FROM matching_pairs
    INNER JOIN concept_id_to_name2
    ON matching_pairs.concept_id = concept_id_to_name2.concept_id
    WHERE name_stripped != concept_name_stripped
""").df()




# add id
matching_pairs['name_id'] = range(1, len(matching_pairs) + 1)

matching_pairs = matching_pairs[['concept_id', 'concept_name','source', 'source_id', 'type', 'name_id', 'name']].reset_index(drop=True)

matching_pairs.to_feather(os.path.join(output_dir, 'matching_pairs.feather'))



tracedf(matching_pairs)
#> DataFrame dimensions: 4131352 rows × 7 columns
#> Column names:
#> ['concept_id', 'concept_name', 'source', 'source_id', 'type', 'name_id', 'name']
#> Estimated memory usage: 1.67 GB



