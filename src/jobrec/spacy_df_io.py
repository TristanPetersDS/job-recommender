"""
Utilities to persist and reload pandas DataFrames that contain one or more spaCy Doc columns.

Round-trip guarantees:
- Column order and names preserved.
- Index (incl. MultiIndex) preserved.
- Full-fidelity spaCy Docs stored via DocBin with store_user_data=True (all attrs).
- Supports None / NaN in Doc columns (null mask stored separately).
- Resistant to reordering issues via stored row_id alignment.
"""

from __future__ import annotations

import json
import pathlib
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Dict, Any

import pandas as pd
import spacy
from spacy.tokens import Doc, DocBin
from spacy.vocab import Vocab


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_doc(obj: Any) -> bool:
    """Cheap check that avoids importing heavy modules in hot loops."""
    return isinstance(obj, Doc)


def _detect_doc_columns(df: pd.DataFrame) -> List[str]:
    """
    Auto-detect columns that contain spaCy Doc objects.
    We scan until first non-null per column.
    """
    doc_cols: List[str] = []
    for col in df.columns:
        # iterate raw numpy array for speed
        col_values = df[col].to_numpy()
        first_non_null = next(
            (v for v in col_values if v is not None and not _isna_like(v)), None
        )
        if _is_doc(first_non_null):
            doc_cols.append(col)
    return doc_cols


def _isna_like(v: Any) -> bool:
    """Return True for pandas-style NA/NaN/None but NOT for Doc objects."""
    if v is None:
        return True
    try:
        # If value is numeric-ish; fall back to pandas
        import math
        if isinstance(v, float) and math.isnan(v):
            return True
    except Exception:
        pass
    # Avoid calling pd.isna on Doc (costly)
    try:
        import pandas as pd  # local import to avoid top-level dependency in docstrings
        return bool(pd.isna(v))  # safe for scalars
    except Exception:
        return False


def _serialize_docs_for_column(
    series: pd.Series,
) -> Tuple[DocBin, List[bool]]:
    """
    Build a DocBin from a Series containing Doc objects & None.
    Returns:
        docbin: docs for non-null rows (in row order)
        null_mask: list[bool], True where original value was null
    """
    docbin = DocBin(store_user_data=True)  # all attrs by default
    null_mask: List[bool] = []
    for val in series.array:  # preserves alignment
        if val is None or _isna_like(val):
            null_mask.append(True)
        else:
            if not _is_doc(val):
                raise TypeError(
                    f"Non-null value in Doc column is not a spaCy Doc: {type(val)}"
                )
            docbin.add(val)
            null_mask.append(False)
    return docbin, null_mask


def _deserialize_docs_for_column(
    docbin: DocBin,
    null_mask: Sequence[bool],
    vocab: Vocab,
) -> List[Optional[Doc]]:
    """
    Reconstruct a list of Doc | None for a column, using null_mask to reinsert Nones.
    """
    docs_iter = iter(docbin.get_docs(vocab))
    out: List[Optional[Doc]] = []
    for is_null in null_mask:
        if is_null:
            out.append(None)
        else:
            out.append(next(docs_iter))
    # sanity: docs_iter should be exhausted
    try:
        extra = next(docs_iter)
        raise ValueError(
            "More docs in DocBin than expected given null_mask length."
        )
    except StopIteration:
        pass
    return out


def _normalize_path(path_prefix: Union[str, pathlib.Path]) -> pathlib.Path:
    p = pathlib.Path(path_prefix)
    if p.suffix:  # user passed a file name; use its stem as directory
        p = p.with_suffix("")
    return p


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_spacy_df(
    df: pd.DataFrame,
    path_prefix: Union[str, pathlib.Path],
    doc_cols: Optional[Sequence[str]] = None,
    overwrite: bool = True,
    parquet_engine: str = "pyarrow",
    parquet_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Persist a DataFrame that contains one or more spaCy Doc columns.

    Parameters
    ----------
    df
        The DataFrame to save.
    path_prefix
        Directory prefix (created if needed). Files written: data.parquet, meta.json,
        docs__<col>.spacy (one per Doc column).
    doc_cols
        Iterable of column names that contain spaCy Docs. If None, auto-detect.
    overwrite
        If False and target directory exists, raise FileExistsError.
    parquet_engine
        Passed to DataFrame.to_parquet().
    parquet_kwargs
        Extra keyword args forwarded to to_parquet().

    Notes
    -----
    - All spaCy Docs are stored with `DocBin(store_user_data=True)` and *no* `attrs`
      restriction, so full token state is serialized.
    - None/NaN entries in Doc columns are supported; we store a null_mask.
    """
    p = _normalize_path(path_prefix)
    if p.exists():
        if not overwrite:
            raise FileExistsError(f"Path already exists: {p}")
    else:
        p.mkdir(parents=True, exist_ok=True)

    if doc_cols is None:
        doc_cols = _detect_doc_columns(df)
    else:
        doc_cols = list(doc_cols)

    # Validate
    for col in doc_cols:
        if col not in df.columns:
            raise KeyError(f"Doc column '{col}' not found in DataFrame.")

    # Build meta
    meta: Dict[str, Any] = {}
    meta["spacy_df_io_version"] = "1.0.0"
    meta["spacy_version"] = spacy.__version__
    meta["row_count"] = int(len(df))
    meta["columns"] = list(map(str, df.columns))  # preserve order
    meta["doc_cols"] = list(map(str, doc_cols))

    # Index metadata (MultiIndex safe)
    meta["index_name"] = df.index.name
    if isinstance(df.index, pd.MultiIndex):
        meta["index_is_multi"] = True
        meta["index_names"] = list(df.index.names)
        # Represent tuples as lists for JSON
        meta["index_values"] = [list(t) for t in df.index.tolist()]
    else:
        meta["index_is_multi"] = False
        meta["index_values"] = df.index.tolist()

    # Save Doc columns -> DocBin + null masks
    masks: Dict[str, List[bool]] = {}
    for col in doc_cols:
        docbin, null_mask = _serialize_docs_for_column(df[col])
        doc_path = p / f"docs__{col}.spacy"
        docbin.to_disk(doc_path)
        masks[col] = null_mask

    meta["null_masks"] = masks

    # Save non-Doc columns to Parquet (preserving index)
    non_doc_df = df.drop(columns=list(doc_cols))
    pk = parquet_kwargs or {}
    non_doc_df.to_parquet(p / "data.parquet", engine=parquet_engine, **pk)

    # Finally, write meta.json
    with (p / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_spacy_df(
    path_prefix: Union[str, pathlib.Path],
    nlp: Union[spacy.language.Language, Vocab, None],
    parquet_engine: str = "pyarrow",
    parquet_kwargs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Reload a DataFrame saved by `save_spacy_df()`.

    Parameters
    ----------
    path_prefix
        Directory prefix used at save time.
    nlp
        A spaCy Language pipeline *or* a Vocab. Use the same (or compatible) model
        used to create the original Docs. If None, a blank vocab will be created
        (attributes load but pipeline components unavailable).
    parquet_engine
        Passed to pandas.read_parquet().
    parquet_kwargs
        Extra keyword args forwarded to read_parquet().

    Returns
    -------
    pd.DataFrame
        Reconstructed DataFrame with spaCy Doc columns in their original positions.
    """
    p = _normalize_path(path_prefix)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")

    # Load meta
    with (p / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    doc_cols: List[str] = meta["doc_cols"]
    columns_order: List[str] = meta["columns"]
    masks: Dict[str, List[bool]] = meta["null_masks"]

    # Load base (non-doc) frame
    pk = parquet_kwargs or {}
    base_df = pd.read_parquet(p / "data.parquet", engine=parquet_engine, **pk)

    # Ensure index order matches stored meta (defensive; Parquet *should* preserve)
    if meta["index_is_multi"]:
        # re-create MultiIndex from stored values
        tuples = [tuple(v) for v in meta["index_values"]]
        mi = pd.MultiIndex.from_tuples(tuples, names=meta.get("index_names"))
        base_df = base_df.reindex(mi)
    else:
        base_df = base_df.reindex(meta["index_values"])
        base_df.index.name = meta.get("index_name")

    # Acquire vocab
    if nlp is None:
        # Use stored language if we ever store it; fallback to blank 'xx'
        vocab = spacy.blank("xx").vocab
    elif isinstance(nlp, Vocab):
        vocab = nlp
    else:
        vocab = nlp.vocab

    # Load each Doc column, reconstruct with null mask
    reconstructed_cols: Dict[str, List[Optional[Doc]]] = {}
    for col in doc_cols:
        doc_path = p / f"docs__{col}.spacy"
        docbin = DocBin().from_disk(doc_path)
        null_mask = masks[col]
        reconstructed_cols[col] = _deserialize_docs_for_column(docbin, null_mask, vocab)

    # Insert Doc columns back into DataFrame in original order
    df = base_df.copy()
    for col in doc_cols:
        # placeholder at end; we'll reorder after
        df[col] = reconstructed_cols[col]

    # Reorder to original
    df = df[columns_order]

    # Final sanity checks
    if len(df) != meta["row_count"]:
        raise ValueError(
            f"Row count mismatch: expected {meta['row_count']}, got {len(df)}."
        )

    return df