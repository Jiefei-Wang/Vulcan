import pandas as pd
import os
from modules.TOKENS import TOKENS
from modules.CodeBlockExecutor import trace, tracedf

omop_base_path = "data/omop_feather"
matching_base_path = "data/matching"
relation_base_path = "data/relation"

if not os.path.exists(relation_base_path):
    os.makedirs(relation_base_path)

target_concepts = pd.read_feather(os.path.join(matching_base_path, 'target_concepts.feather'))
concept_ancestors = pd.read_feather(os.path.join(omop_base_path, 'concept_ancestor.feather'))

# We want both ancestor and descendant to be in target_concepts
relation_tables = concept_ancestors[
    concept_ancestors['ancestor_concept_id'].isin(target_concepts['concept_id'])&
    concept_ancestors['descendant_concept_id'].isin(target_concepts['concept_id'])&
    (concept_ancestors['min_levels_of_separation'] >= 1) &
    (concept_ancestors['min_levels_of_separation'] <= 2)
    ].drop_duplicates(subset=['ancestor_concept_id', 'descendant_concept_id']).reset_index(drop=True)


name_bridge_relation = relation_tables.rename(
    columns={
        'ancestor_concept_id': 'concept_id',
        'descendant_concept_id': 'name_id'
    }
    )[[ 'concept_id', 'name_id']].reset_index(drop=True)

name_table_relation = target_concepts.rename(
    columns={
        'concept_id': 'name_id',
        'concept_name': 'name'
    }
    )[[ 'name_id', 'name']]

# filter out the names that are not in the name_bridge_relation
name_table_relation = name_table_relation[
    name_table_relation.name_id.isin(name_bridge_relation['name_id'])].reset_index(drop=True)

name_table_relation['source'] = "OMOP"
name_table_relation['source_id'] = name_table_relation['name_id']
name_table_relation['type'] = "child"

name_table_relation['name'] = TOKENS.parent + name_table_relation['name'] 


name_table_relation = name_table_relation.reset_index(drop=True)


name_table_relation.to_feather(os.path.join(relation_base_path, 'name_table_relation.feather'))
name_bridge_relation.to_feather(os.path.join(relation_base_path, 'name_bridge_relation.feather'))

tracedf(name_table_relation)
#> DataFrame dimensions: 159646 rows × 5 columns
#> Column names:
#> ['name_id', 'name', 'source', 'source_id', 'type']
#> Estimated memory usage: 34.79 MB

tracedf(name_bridge_relation)
#> DataFrame dimensions: 1029979 rows × 2 columns
#> Column names:
#> ['concept_id', 'name_id']
#> Estimated memory usage: 15.72 MB

trace(len(name_bridge_relation['concept_id'].unique()))
#> 42099

# find name_id with most concept_id
name_id_counts = name_bridge_relation['name_id'].value_counts().reset_index()
name_id_counts.columns = ['name_id', 'concept_count']

trace(name_id_counts)
#>          name_id  concept_count
#> 0       37116355             45
#> 1       36713732             43
#> 2       37118954             42
#> 3       35622345             42
#> 4       37116269             41
#> ...          ...            ...
#> 159641   4219711              1
#> 159642   4311981              1
#> 159643   4161937              1
#> 159644   4011770              1
#> 159645  37168922              1
#> 
#> [159646 rows x 2 columns]

trace(relation_tables[relation_tables.descendant_concept_id==name_id_counts.name_id[0]])
#>         ancestor_concept_id  descendant_concept_id  min_levels_of_separation  max_levels_of_separation
#> 514613               441953               37116355                         1                         1
#> 517058               134741               37116355                         2                         5
#> 519319               372424               37116355                         2                         2
#> 520181               436805               37116355                         1                         1
#> 522952              4077761               37116355                         2                         2
#> 524138              4134422               37116355                         2                         4
#> 524306              4149616               37116355                         1                         1
#> 526314              4312226               37116355                         1                         1
#> 530811              4180651               37116355                         1                         1
#> 531400             37018424               37116355                         2                         2
#> 536792               380378               37116355                         1                         1
#> 537835              4027258               37116355                         2                         2
#> 541360              4132553               37116355                         2                         3
#> 542103              4190397               37116355                         2                         2
#> 542145              4197328               37116355                         2                         4
#> 544219               435244               37116355                         2                         7
#> 545839               443916               37116355                         2                         3
#> 550459               381854               37116355                         2                         3
#> 550559               435645               37116355                         1                         1
#> 554556              4051331               37116355                         1                         1
#> 555206              4051577               37116355                         1                         1
#> 560372              4337941               37116355                         2                         4
#> 561257             40277917               37116355                         1                         1
#> 565079             37164426               37116355                         2                         2
#> 572209              4029498               37116355                         2                         2
#> 574811              4134440               37116355                         2                         7
#> 576740              4183733               37116355                         2                         2
#> 590125              4180314               37116355                         1                         1
#> 595106              4201554               37116355                         2                         2
#> 598547               376337               37116355                         2                         5
#> 599207               381860               37116355                         2                         2
#> 600882               440409               37116355                         2                         4
#> 601928              4043345               37116355                         2                         4
#> 603335              4181326               37116355                         1                         1
#> 606297              4274970               37116355                         2                         2
#> 606948              4334739               37116355                         1                         1
#> 608586              4077967               37116355                         2                         2
#> 610412              4145142               37116355                         2                         3
#> 620217              4301416               37116355                         2                         4
#> 621440              4344497               37116355                         2                         2
#> 623835             46269998               37116355                         1                         1
#> 625154               443432               37116355                         2                         2
#> 626686             45771096               37116355                         2                         2
#> 633274              4180158               37116355                         2                         2
#> 637631             37396727               37116355                         2                         2


