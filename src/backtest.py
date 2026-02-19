"""
Walk-forward / rolling backtest utilities.

This module generates time-series splits (train/val/test) without leakage.

Key idea:
- Each fold evaluates a different out-of-sample segment.
- Training always happens strictly before validation and test.
- Splits are index-based to keep it lightweight and source-agnostic.

We intentionally do NOT import TensorFlow/Keras here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class FoldSplit:
    """
    Represents a single walk-forward split using integer indices.

    All ranges follow Python slicing semantics: [start, end)
    """
    fold_id: int

    train_start: int
    train_end: int

    val_start: int
    val_end: int

    test_start: int
    test_end: int

    def as_dict(self) -> dict:
        """Serialize split ranges for metadata logging."""
        return {
            "fold_id": self.fold_id,
            "train": [self.train_start, self.train_end],
            "val": [self.val_start, self.val_end],
            "test": [self.test_start, self.test_end],
        }


def generate_walk_forward_splits(
    n_rows: int,
    *,
    n_folds: int = 3,
    test_size: int = 252,
    val_size: int = 126,
    min_train_size: int = 756,
) -> List[FoldSplit]:
    """
    Generate walk-forward splits anchored at the end of the dataset.

    Example (n_folds=3, test_size=252):
    - Fold 1 tests the oldest chunk among the last 3 test windows
    - Fold 3 tests the most recent chunk

    Layout for each fold (chronological):
      [ train ............ ][ val .... ][ test .... ]

    Parameters
    ----------
    n_rows : int
        Total number of rows available after feature engineering and dropna.
        (We split after features are created to avoid dealing with NaNs here.)
    n_folds : int
        Number of walk-forward folds (default 3).
    test_size : int
        Size of each test window in rows (default 252 trading days).
    val_size : int
        Size of validation window immediately before test (default 126).
    min_train_size : int
        Minimum number of rows required for training (default 756 ~ 3 years).

    Returns
    -------
    list[FoldSplit]
        List of splits ordered from oldest test window to most recent.

    Raises
    ------
    ValueError
        If there isn't enough data to build the requested folds.
    """
    if n_rows <= 0:
        raise ValueError("n_rows must be positive.")

    total_required = n_folds * test_size + n_folds * val_size + min_train_size
    if n_rows < total_required:
        raise ValueError(
            f"Not enough rows for walk-forward splits. "
            f"Need >= {total_required}, got {n_rows}. "
            f"(n_folds={n_folds}, test_size={test_size}, val_size={val_size}, min_train_size={min_train_size})"
        )

    splits: List[FoldSplit] = []

    # Anchor at the end: the last fold tests the most recent period.
    # We build folds backwards and then reverse to keep chronological order.
    test_end = n_rows

    for fold_idx in range(n_folds, 0, -1):
        test_start = test_end - test_size
        val_end = test_start
        val_start = val_end - val_size

        # Train starts at 0 for simplicity in the "anchored" scheme.
        # You can later evolve this to use rolling train windows.
        train_start = 0
        train_end = val_start

        if train_end - train_start < min_train_size:
            raise ValueError(
                f"Fold {fold_idx}: train window too small "
                f"({train_end - train_start} < {min_train_size})."
            )

        splits.append(
            FoldSplit(
                fold_id=fold_idx, # temporary id (will be renumbered after reverse)
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

        # Move anchor backwards for next fold
        test_end = test_start

    # Oldest fold first
    splits.reverse()

    # Renumber folds in chronological order: 1 = oldest test window
    splits = [
        FoldSplit(
            fold_id=i + 1,
            train_start=s.train_start,
            train_end=s.train_end,
            val_start=s.val_start,
            val_end=s.val_end,
            test_start=s.test_start,
            test_end=s.test_end,
        )
        for i, s in enumerate(splits)
    ]
    
    return splits
