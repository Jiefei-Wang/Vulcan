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
map_tables = [
    'map_table_umls.feather',
    'map_table_OMOP.feather',
]

####################
## Combine all map_tables
####################
combined_mapping_table = pd.concat(
    [pd.read_feather(os.path.join(input_dir, name_map_table)) for name_map_table in map_tables],
    ignore_index=True
)
combined_mapping_table['source_id'] = combined_mapping_table['source_id'].astype(str)

trace(combined_mapping_table.shape)
#> (11370225, 5)

# - For each concept_id in combined_mapping_table, keep only the concepts that are in the standard bridge
#   1. if the concept_id is standard, it is in the std_bridge
#   2. if the concept_id is non-standard, it will be mapped to a standard concept_id in the std_bridge
#   3. For those that are not in the std_bridge, there is no way to map them 
# to a standard concept, so we will not use them in the training.
# - For name_stripped, keep only letters in name, all lowercase from name
# - Remove empty names
matching_map_table = duckdb.query("""
    SELECT std_bridge.std_concept_id AS concept_id, source, source_id, type, name,
    LOWER(REGEXP_REPLACE(name, '[^a-zA-Z0-9]', '', 'g')) AS name_stripped
    FROM combined_mapping_table
    inner join std_bridge
    ON combined_mapping_table.concept_id = std_bridge.concept_id
    where name IS NOT NULL AND name != ''
""").df()

trace(matching_map_table.shape)
#> (9007100, 6)


## remove non-english rows like: 인도신1mg주
matching_map_table = matching_map_table[matching_map_table['name'].apply(lambda x: x.isascii())].reset_index(drop=True)

trace(matching_map_table.shape)
#> (8152148, 6)

## for each concept_id, remove the duplicates in name_stripped
matching_map_table = duckdb.query("""
    SELECT *
    FROM (
        SELECT *, 
            ROW_NUMBER() OVER (PARTITION BY concept_id, name_stripped ORDER BY source_id) AS rn
        FROM matching_map_table
    ) t
    WHERE rn = 1;
"""
).df()



matching_map_table = matching_map_table[['concept_id', 'source', 'source_id', 'type', 'name']].reset_index(drop=True)

matching_map_table.to_feather(os.path.join(output_dir, 'matching_map_table.feather'))


tracedf(matching_map_table)
#> DataFrame dimensions: 5696264 rows × 5 columns
#> Column names:
#> ['concept_id', 'source', 'source_id', 'type', 'name']
#> Estimated memory usage: 1.53 GB



