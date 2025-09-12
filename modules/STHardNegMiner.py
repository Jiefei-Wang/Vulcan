from typing import List, Optional
import pandas as pd
from datasets import Dataset
from sentence_transformers.util import mine_hard_negatives
from sentence_transformers import SentenceTransformer


def _pairs_to_dataset(pairs_df: pd.DataFrame) -> Dataset:
    # Ensure only needed cols and drop dups
    df = pairs_df[["anchor", "positive"]].drop_duplicates().reset_index(drop=True)
    return Dataset.from_pandas(df, preserve_index=False)


def _expand_negatives_to_pairs(mined: Dataset) -> pd.DataFrame:
    cols = mined.column_names
    df = mined.to_pandas()
    # Identify negative columns (negative or negative_1 ...)
    neg_cols = [c for c in cols if c == "negative" or c.startswith("negative_")]
    if not neg_cols:
        # No negatives mined; return empty
        return pd.DataFrame(columns=["sentence1", "sentence2", "label"])
    # Melt negatives into rows
    long = df.melt(id_vars=["anchor", "positive"], value_vars=neg_cols, var_name="neg_col", value_name="negative")
    long = long.dropna(subset=["negative"])  # drop missing
    # Map to training pair schema
    out = pd.DataFrame({
        "sentence1": long["negative"],  # corpus negative text
        "sentence2": long["anchor"],    # query text
        "label": 0,
    })
    return out.reset_index(drop=True)


def mine_negatives(
    *,
    anchor_positive_pairs: pd.DataFrame,
    model: SentenceTransformer,
    range_min: int = 10,
    range_max: Optional[int] = 50,
    margin: Optional[float] = 0.05,
    num_negatives: int = 5,
    sampling_strategy: str = "random",
    batch_size: int = 128,
    use_faiss: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Use sentence_transformers.util.mine_hard_negatives to create negative pairs
    for training with ContrastiveLoss.

    anchor_positive_pairs: DataFrame with columns ['anchor','positive']
    Returns: DataFrame with columns ['sentence1','sentence2','label']
    """
    ds = _pairs_to_dataset(anchor_positive_pairs)
    mined = mine_hard_negatives(
        dataset=ds,
        model=model,
        range_min=range_min,
        range_max=range_max,
        margin=margin,
        num_negatives=num_negatives,
        sampling_strategy=sampling_strategy,
        batch_size=batch_size,
        use_faiss=use_faiss,
        verbose=verbose,
    )
    return _expand_negatives_to_pairs(mined)

