"""The model pipeline: video frames in, FrameResult objects out.

Single-file on purpose. Everything you need to debug the model lives here.
No JSON, no run directories, no CLI - just functions you can call from a
notebook and inspect.

Usage from a notebook:

    from soccer_vision.pipeline import Pipeline
    pipe = Pipeline()
    for result in pipe.run("clip.mp4", max_frames=300):
        print(result.frame_index, len(result.players), result.ball is not None)
"""
from __future__ import annotations

from typing import Iterator, Optional

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from .types import (
    Detection,
    FrameResult,
    ProximityEvent,
    TrackedBall,
    TrackedPlayer,
)


# COCO class indices we care about
COCO_PERSON = 0
COCO_SPORTS_BALL = 32


class Pipeline:
    """Detection + tracking + simple proximity heuristic.

    Defaults are tuned for CPU on a laptop with a short clip. Tweak from
    a notebook by passing kwargs - everything is exposed.
    """

    def __init__(
        self,
        yolo_model: str = "yolov8n.pt",
        detection_confidence: float = 0.30,
        ball_detection_confidence: float = 0.10,
        sample_every_n: int = 3,
        proximity_threshold_ratio: float = 0.6,
        tracker_match_threshold: float = 0.80,
    ) -> None:
        # Detection
        self.model = YOLO(yolo_model)
        self.detection_confidence = detection_confidence
        self.ball_detection_confidence = ball_detection_confidence
        self.sample_every_n = sample_every_n

        # Tracker - one for players. The ball gets handled separately because
        # it has very different motion characteristics and the IoU-based
        # tracker confuses ball detections with each other across frames.
        self.player_tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=tracker_match_threshold,
            frame_rate=30 // sample_every_n,
        )

        # Proximity heuristic: a player and ball are "close" if the distance
        # between their centers is less than this fraction of the player's
        # bbox diagonal. Larger -> more permissive. 0.6 is a starting guess.
        self.proximity_threshold_ratio = proximity_threshold_ratio

    # ------------------------------------------------------------------
    # Stage 1: detection
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run YOLO on one BGR frame and return Detection objects.

        We use a lower confidence for the ball than for players because the
        ball is small, often blurred, and YOLO is unsure about it more often.
        """
        # Use the lower of the two thresholds when calling YOLO; we'll filter
        # players above the higher threshold afterwards.
        min_conf = min(self.detection_confidence, self.ball_detection_confidence)
        results = self.model.predict(
            source=frame,
            conf=min_conf,
            iou=0.50,
            classes=[COCO_PERSON, COCO_SPORTS_BALL],
            max_det=50,
            verbose=False,
        )[0]

        detections: list[Detection] = []
        if results.boxes is None or len(results.boxes) == 0:
            return detections

        boxes = results.boxes
        for i in range(len(boxes)):
            cls = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            xyxy = tuple(float(v) for v in boxes.xyxy[i].tolist())

            if cls == COCO_PERSON and conf >= self.detection_confidence:
                detections.append(Detection(
                    bbox_xyxy=xyxy, confidence=conf, class_name="player"
                ))
            elif cls == COCO_SPORTS_BALL and conf >= self.ball_detection_confidence:
                detections.append(Detection(
                    bbox_xyxy=xyxy, confidence=conf, class_name="ball"
                ))

        return detections

    # ------------------------------------------------------------------
    # Stage 2: tracking
    # ------------------------------------------------------------------

    def track_players(
        self, detections: list[Detection]
    ) -> list[TrackedPlayer]:
        """Pass player detections through ByteTrack to get persistent IDs."""
        player_dets = [d for d in detections if d.class_name == "player"]
        if not player_dets:
            return []

        # Convert to supervision.Detections for ByteTrack
        sv_dets = sv.Detections(
            xyxy=np.array([d.bbox_xyxy for d in player_dets], dtype=np.float32),
            confidence=np.array([d.confidence for d in player_dets], dtype=np.float32),
            class_id=np.zeros(len(player_dets), dtype=int),
        )
        tracked = self.player_tracker.update_with_detections(sv_dets)

        results: list[TrackedPlayer] = []
        if tracked.tracker_id is None:
            return results
        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i])
            if tid < 0:
                continue
            results.append(TrackedPlayer(
                track_id=f"p_{tid:03d}",
                bbox_xyxy=tuple(float(v) for v in tracked.xyxy[i].tolist()),
                confidence=float(tracked.confidence[i]) if tracked.confidence is not None else 0.0,
            ))
        return results

    @staticmethod
    def select_ball(detections: list[Detection]) -> Optional[TrackedBall]:
        """Pick the highest-confidence ball detection, if any.

        We don't track the ball with ByteTrack - there's only ever one and
        the matching algorithm doesn't help. Better ball tracking needs a
        specialized model (TrackNet) which is a v0.4+ task.
        """
        ball_dets = [d for d in detections if d.class_name == "ball"]
        if not ball_dets:
            return None
        best = max(ball_dets, key=lambda d: d.confidence)
        return TrackedBall(bbox_xyxy=best.bbox_xyxy, confidence=best.confidence)

    # ------------------------------------------------------------------
    # Stage 3: ball-proximity heuristic
    # ------------------------------------------------------------------

    def compute_proximity(
        self,
        players: list[TrackedPlayer],
        ball: Optional[TrackedBall],
    ) -> list[ProximityEvent]:
        """For each player, decide whether the ball is 'close enough'.

        Uses two signals:
          1. Distance between player center and ball center, normalized by
             the player's bbox diagonal (handles different zoom levels).
          2. Whether the bounding boxes actually overlap (sharper signal but
             often misses contact when bboxes are tight).

        We return an event for every player whose normalized distance is
        under the threshold. The 'nearest_player_to_ball' on FrameResult
        picks the closest one.
        """
        if ball is None or not players:
            return []

        bx, by = ball.center
        events: list[ProximityEvent] = []
        for p in players:
            px, py = p.center
            dist = ((px - bx) ** 2 + (py - by) ** 2) ** 0.5
            diag = p.diagonal
            normalized = dist / diag if diag > 0 else float("inf")

            intersects = _bboxes_intersect(p.bbox_xyxy, ball.bbox_xyxy)

            if normalized < self.proximity_threshold_ratio or intersects:
                events.append(ProximityEvent(
                    track_id=p.track_id,
                    distance_px=dist,
                    player_diagonal_px=diag,
                    bbox_intersects=intersects,
                ))

        return events

    # ------------------------------------------------------------------
    # Stage 4: orchestration
    # ------------------------------------------------------------------

    def process_frame(
        self, frame: np.ndarray, frame_index: int, timestamp_ms: int
    ) -> FrameResult:
        """Run the full pipeline on one frame and return a FrameResult."""
        raw = self.detect(frame)
        players = self.track_players(raw)
        ball = self.select_ball(raw)
        proximity = self.compute_proximity(players, ball)
        return FrameResult(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            raw_detections=raw,
            players=players,
            ball=ball,
            proximity_events=proximity,
        )

    def run(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
    ) -> Iterator[tuple[np.ndarray, FrameResult]]:
        """Iterate over (frame_image, FrameResult) for each sampled frame.

        Yielding the frame image alongside the result is what lets the
        notebook draw bounding boxes on top of it.

        max_frames lets you cut a long video short for fast iteration.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {video_path}")

        try:
            native_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
            sampled_count = 0
            frame_index = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % self.sample_every_n == 0:
                    timestamp_ms = int(round(frame_index / native_fps * 1000))
                    result = self.process_frame(frame, frame_index, timestamp_ms)
                    yield frame, result
                    sampled_count += 1
                    if max_frames is not None and sampled_count >= max_frames:
                        break
                frame_index += 1
        finally:
            cap.release()


def _bboxes_intersect(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    """True if two xyxy bounding boxes overlap by even one pixel."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)
