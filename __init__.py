"""Soccer vision prototype - pure Python ML pipeline.

Pipeline is not eagerly imported because it pulls in ultralytics + PyTorch,
which is slow. Import it explicitly when needed:

    from soccer_vision.pipeline import Pipeline
    from soccer_vision.types import FrameResult
    from soccer_vision.viz import draw_overlay, render_to_mp4
"""
