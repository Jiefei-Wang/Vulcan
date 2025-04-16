## magic commands to tell ide I have the following variables defined
import pandas as pd

concept: pd.DataFrame  # DataFrame loaded from 'concept.feather'
concept_relationship: pd.DataFrame  # DataFrame loaded from 'concept_relationship.feather'
concept_ancestor: pd.DataFrame  # DataFrame loaded from 'concept_ancestor.feather'
conceptML: pd.DataFrame  # DataFrame loaded from 'conceptML.feather'

std_target: pd.DataFrame  # Target standard concepts filtered from conceptML
reserved_concepts: pd.DataFrame  # Reserved concepts filtered from concept
reserved_vocab: str  # Vocabulary ID for reserved concepts
