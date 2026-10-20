"""Phase 4 capture: camera card detections + manual action logs -> schema hand histories.

TODO(Phase 4): card detector (camera -> ``CardDetection``), action-entry app (-> ``ManualHandLog``),
and a live recorder that writes in-progress hands through ``HandHistoryStore`` so hole cards
stay sealed until each hand completes.
"""

from poker_lab.capture.ingestor import (
    CaptureIngestor,
    CardDetection,
    Conflict,
    ConflictKind,
    IngestResult,
    ManualAction,
    ManualCardEntry,
    ManualHandLog,
)

__all__ = [
    "CaptureIngestor",
    "CardDetection",
    "Conflict",
    "ConflictKind",
    "IngestResult",
    "ManualAction",
    "ManualCardEntry",
    "ManualHandLog",
]
