"""Batch import of completed hand histories from text files (P3.6), offline only.

``phh``: the open PHH format (Kim 2024), strictly replayed through PokerKit's rules, with an
exporter for round trips. ``python -m poker_lab.importers`` converts files to JSON Lines.
"""

from poker_lab.importers.phh import (
    ImportRejection,
    ImportReport,
    PHHImportError,
    RejectReason,
    as_imported,
    dumps_phh,
    hand_from_phh,
    hand_to_phh,
    import_phh,
    loads_phh,
    phh_files,
    read_phh,
    write_phh,
)

__all__ = [
    "ImportRejection",
    "ImportReport",
    "PHHImportError",
    "RejectReason",
    "as_imported",
    "dumps_phh",
    "hand_from_phh",
    "hand_to_phh",
    "import_phh",
    "loads_phh",
    "phh_files",
    "read_phh",
    "write_phh",
]
