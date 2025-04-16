
import pandas as pd


def read_mrconso(mrconso_path):
    mrconso_columns = [
        "CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", "SCUI",
        "SDUI", "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS", "CVF"
    ]
    mrconso_path = "data/UMLS_raw/MRCONSO.RRF"
    mrconso_df = pd.read_csv(
        mrconso_path, delimiter="|", names=mrconso_columns, dtype=str, header=None, index_col=False
    )
    # Drop the last empty column caused by the trailing delimiter
    mrconso_df = mrconso_df.drop(columns=[mrconso_df.columns[-1]])
    return mrconso_df


# Load UMLS MRDEF.RRF (definitions)
def read_mrdef(umls_def_path):
    umls_def_columns = ["CUI", "AUI", "ATUI", "SATUI", "SAB", "DEF", "SUPPRESS", "CVF"]
    UMLS_def = pd.read_csv(
        umls_def_path, delimiter="|", names=umls_def_columns, dtype=str, header=None, index_col=False
    )
    # Drop the last empty column
    UMLS_def = UMLS_def.drop(columns=[UMLS_def.columns[-1]])
    return UMLS_def


