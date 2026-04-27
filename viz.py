"""Visualization helpers for use in notebooks.

Draws bounding boxes on frames and provides a Jupyter video viewer so you
can watch the model's output frame by frame.
"""
from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from .types import FrameResult, TrackedBall, TrackedPlayer


# BGR (OpenCV) colors
COLOR_PLAYER = (0, 200, 0)        # green
COLOR_BALL = (0, 165, 255)        # orange
COLOR_PROXIMITY = (0, 0, 255)     # red


def draw_overlay(frame: np.ndarray, result: FrameResult) -> np.ndarray:
    """Draw bounding boxes and labels on a copy of the frame.

    - Players: green box with track ID
    - Ball: orange box
    - Player who is nearest to ball (if any): red box (overwrites green)
    """
    out = frame.copy()

    # Find which player (if any) is the nearest-to-ball this frame
    nearest_id: str | None = None
    nearest = result.nearest_player_to_ball
    if nearest is not None:
        nearest_id = nearest.track_id

    for player in result.players:
        color = COLOR_PROXIMITY if player.track_id == nearest_id else COLOR_PLAYER
        _draw_box(out, player.bbox_xyxy, color, label=player.track_id)

    if result.ball is not None:
        _draw_box(out, result.ball.bbox_xyxy, COLOR_BALL,
                  label=f"ball {result.ball.confidence:.2f}")

    # Status bar in the top-left
    status = (
        f"frame {result.frame_index}  "
        f"t={result.timestamp_ms}ms  "
        f"players={len(result.players)}  "
        f"ball={'yes' if result.ball else 'no'}  "
        f"nearest={nearest_id or '-'}"
    )
    cv2.putText(out, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _draw_box(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    color: tuple[int, int, int],
    label: str | None = None,
) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        # Label background for readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def render_to_mp4(
    frames_and_results: Iterable[tuple[np.ndarray, FrameResult]],
    output_path: str,
    fps: int = 10,
) -> str:
    """Run the pipeline and save the annotated video to disk.

    Use this when you want to scrub through the result with VLC or share
    it with someone. For interactive viewing in a notebook, use
    `display_in_notebook` below.
    """
    writer: cv2.VideoWriter | None = None
    count = 0
    for frame, result in frames_and_results:
        annotated = draw_overlay(frame, result)
        if writer is None:
            h, w = annotated.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        writer.write(annotated)
        count += 1
    if writer is not None:
        writer.release()
    return f"Wrote {count} frames to {output_path}"
