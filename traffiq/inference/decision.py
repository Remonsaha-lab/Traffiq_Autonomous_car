"""
============================================================
 TRAFFIQ v2 — Decision Layer
 File: inference/decision_layer.py

 This is the BRAIN of the car during runtime.

 It takes two inputs:
   1. CNN prediction → raw [speed, direction] from the
      neural network looking at the full image
   2. OpenCV results → structured data about line position
      and obstacle locations from cv_pipeline.py

 And produces ONE final output:
   [speed, direction] — both clipped to [-1, 1]

 WHY HAVE A SEPARATE DECISION LAYER?
   The CNN is trained on smooth driving data. It has no
   explicit knowledge of:
     - "There is an obstacle 30cm ahead — STOP"
     - "I cannot see the line — slow down"
     - "Obstacle is to my left — steer right"

   The CNN implicitly learns some of this from data, but
   the OpenCV pipeline gives us explicit, reliable signals
   for these safety-critical situations.

   The decision layer combines both:
   - Normal driving → CNN controls everything
   - Obstacle detected → override speed, blend steering
   - Line lost → slow down, hold last direction
   - Emergency → full stop
============================================================
"""

import numpy as np


class DecisionLayer:
    """
    Merges CNN output with OpenCV features to produce
    the final [speed, direction] command.

    State is maintained across frames:
      - last_direction: remembered when line is lost
      - frames_since_line: counts how long line was missing
    """

    def __init__(self):
        self.last_direction     = 0.0
        self.frames_since_line  = 0
        self.LINE_LOST_PATIENCE = 10   # frames before we slow significantly

    def decide(
        self,
        cnn_speed:     float,
        cnn_direction: float,
        line:          dict,
        obstacles:     dict
    ) -> tuple:
        """
        Main decision function. Called every frame.

        Parameters:
          cnn_speed     : float in [-1, 1] from CNN
          cnn_direction : float in [-1, 1] from CNN
          line          : dict from detect_white_line()
          obstacles     : dict from detect_obstacles()

        Returns:
          (final_speed, final_direction) — both in [-1, 1]
        """

        speed     = cnn_speed
        direction = cnn_direction

        # ── STAGE 1: Line tracking ────────────────────────
        # If the line is detected, update memory and use
        # a blend of CNN direction and OpenCV line offset.
        # If line is lost, rely on memory + slow down.

        if line['found']:
            self.frames_since_line = 0
            self.last_direction    = cnn_direction

            # Blend CNN direction with raw OpenCV offset
            # CNN: learned from full image context (handles curves well)
            # OpenCV offset: direct measurement (handles drift well)
            # 70% CNN, 30% direct measurement
            blended_direction = 0.7 * cnn_direction + 0.3 * line['offset']
            direction = blended_direction

        else:
            # Line not found — count consecutive lost frames
            self.frames_since_line += 1

            # Hold last known direction
            direction = self.last_direction

            # Slow down proportionally to how long line has been lost
            # At 10 frames lost: speed reduced to 30% of CNN suggestion
            lost_factor = max(
                0.3,
                1.0 - self.frames_since_line / self.LINE_LOST_PATIENCE
            )
            speed = cnn_speed * lost_factor

        # ── STAGE 2: Obstacle response ────────────────────
        # Overrides the direction and speed calculated above
        # based on where obstacles are.

        if obstacles['found'] and obstacles['nearest'] is not None:
            nearest   = obstacles['nearest']
            area      = nearest['area_norm']   # 0=tiny/far, 1=huge/close
            side      = nearest['side']        # 'left', 'center', 'right'
            cy_norm   = nearest['cy_norm']     # 1.0 = bottom = very close

            # ── Emergency stop ────────────────────────────
            # Large obstacle directly ahead and very close
            if area > 0.20 and side == 'center' and cy_norm > 0.7:
                return 0.0, direction   # FULL STOP, hold direction

            # ── Strong avoidance ──────────────────────────
            # Moderate obstacle in center
            if area > 0.08 and side == 'center':
                # Determine which side has more space
                # by checking if line offset suggests space
                if line['found']:
                    # Steer away from obstacle, toward line
                    avoidance_dir = -np.sign(line['offset'])
                else:
                    # No line info — steer right as default
                    avoidance_dir = 1.0

                # Blend strength increases with obstacle size
                # area=0.08 → 40% avoidance
                # area=0.20 → full avoidance
                blend     = min((area - 0.08) / 0.12, 1.0)
                direction = (1 - blend) * direction + blend * avoidance_dir

                # Also slow down
                speed = speed * (1.0 - blend * 0.6)

            # ── Soft avoidance ────────────────────────────
            # Obstacle on one side — gentle correction
            elif area > 0.04:
                if side == 'left':
                    # Obstacle left → nudge right
                    correction = 0.3 * area / 0.08
                    direction  = direction + correction
                elif side == 'right':
                    # Obstacle right → nudge left
                    correction = 0.3 * area / 0.08
                    direction  = direction - correction

                # Minor speed reduction
                speed = speed * 0.85

        # ── STAGE 3: Final clipping ───────────────────────
        # Both values MUST be in [-1, 1] as per spec.
        final_speed     = float(np.clip(speed,     -1.0, 1.0))
        final_direction = float(np.clip(direction, -1.0, 1.0))

        return final_speed, final_direction

    def reset(self):
        """Call this before each run starts."""
        self.last_direction    = 0.0
        self.frames_since_line = 0