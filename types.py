"""Plain Python data types for what the pipeline produces.

These dataclasses are the "API" of the model. Code that consumes pipeline
output (notebooks, future JSON writers, future evaluators) all consume these.
Keep them dependency-light so they can be imported anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Detection:
    """One bounding box from the detector before tracking."""
    bbox_xyxy: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    confidence: float
    class_name: str  # "player" or "ball"

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width(self) -> float:
        return self.bbox_xyxy[2] - self.bbox_xyxy[0]

    @property
    def height(self) -> float:
        return self.bbox_xyxy[3] - self.bbox_xyxy[1]

    @property
    def diagonal(self) -> float:
        return (self.width**2 + self.height**2) ** 0.5


@dataclass
class TrackedPlayer:
    """A player detection that the tracker has linked to a persistent ID."""
    track_id: str  # e.g. "p_007"
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def diagonal(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


@dataclass
class TrackedBall:
    """A ball detection. We don't bother with persistent ID for the ball -
    there's only one and tracking it across gaps is its own problem."""
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class ProximityEvent:
    """A frame where the ball is close enough to a player to *possibly* be a touch.

    This is intentionally NOT called 'Touch'. With pixel-space bboxes and no
    pose data, we can detect 'ball is near this player', not 'this player
    touched the ball'. Real touch detection needs pitch homography and ideally
    pose estimation. Treat this as a possession proxy, not ground truth.
    """
    track_id: str
    distance_px: float  # distance between ball center and player center
    player_diagonal_px: float  # for normalizing across zoom levels
    bbox_intersects: bool  # True if bboxes actually overlap


@dataclass
class FrameResult:
    """Everything the pipeline knows about one processed frame."""
    frame_index: int  # index in the original (un-sampled) video
    timestamp_ms: int
    raw_detections: list[Detection] = field(default_factory=list)
    players: list[TrackedPlayer] = field(default_factory=list)
    ball: Optional[TrackedBall] = None
    proximity_events: list[ProximityEvent] = field(default_factory=list)

    @property
    def nearest_player_to_ball(self) -> Optional[ProximityEvent]:
        """The proximity event with the smallest distance, or None."""
        if not self.proximity_events:
            return None
        return min(self.proximity_events, key=lambda e: e.distance_px)
