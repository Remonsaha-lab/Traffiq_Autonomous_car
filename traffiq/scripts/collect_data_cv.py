"""
============================================================
 TRAFFIQ v2 — Data Collection Script
 File: scripts/collect_data_v2.py

 KEY CHANGE FROM v1:
   Records BOTH speed and direction during driving.
   The old script hardcoded throttle at 0.3.
   Now you must vary your speed deliberately:
     - Slow before turns and obstacles
     - Normal speed on clear straights
     - Near-stop when something is directly ahead

 The model learns speed control from watching what
 YOU did — your driving quality directly determines
 how well the model learns to control speed.

 Controls:
   W / ↑     → Throttle forward (hold for more speed)
   S / ↓     → Brake / slow down
   A / ←     → Steer left
   D / →     → Steer right
   R         → Start/Stop recording
   Q         → Quit and save
============================================================
"""

import os
import cv2
import json
import time
import numpy as np
import pygame
from datetime import datetime
from pathlib import Path

# ─── CONFIGURATION ────────────────────────────────────────
SIM_HOST         = "127.0.0.1"
SIM_PORT         = 9091
TARGET_IMAGES    = 6000
SAVE_DIR         = Path("dataset") / datetime.now().strftime("%Y%m%d_%H%M%S")
STEERING_SCALE   = 1.0
THROTTLE_MAX     = 0.6    # Maximum forward speed during collection
THROTTLE_STEP    = 0.05   # How quickly speed ramps up per frame
# ──────────────────────────────────────────────────────────


class DataCollectorV2:
    def __init__(self):
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        (SAVE_DIR / "images").mkdir(exist_ok=True)

        self.frame_count  = 0
        self.record_count = 0
        self.recording    = False
        self.direction    = 0.0
        self.speed        = 0.0          # Now a state variable, not fixed
        self.log          = []

        pygame.init()
        self.screen = pygame.display.set_mode((500, 220))
        pygame.display.set_caption("TRAFFIQ v2 — Data Collector")
        self.font  = pygame.font.SysFont("monospace", 16)
        self.clock = pygame.time.Clock()

        print(f"\n[v2 Collector] Save dir: {SAVE_DIR}")
        print("[v2 Collector] IMPORTANT: Vary your speed!")
        print("               Slow for turns, faster on straights.")
        print("[v2 Collector] Press R to record, Q to quit.\n")

    def get_keyboard_input(self):
        """
        Reads keyboard and updates self.direction and self.speed.
        Speed ramps up gradually when W is held — not instant.
        """
        quit_flag = False
        keys = pygame.key.get_pressed()

        # Direction (instant response)
        if   keys[pygame.K_LEFT]  or keys[pygame.K_a]:
            self.direction = -STEERING_SCALE
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.direction =  STEERING_SCALE
        else:
            # Gradually return to center
            self.direction *= 0.7

        # Speed (gradual ramp — mimics real throttle response)
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            # Ramp up toward max
            self.speed = min(self.speed + THROTTLE_STEP, THROTTLE_MAX)
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            # Ramp down toward zero (braking)
            self.speed = max(self.speed - THROTTLE_STEP * 2, 0.0)
        else:
            # Gentle coast down
            self.speed = max(self.speed - THROTTLE_STEP * 0.5, 0.0)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_flag = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.recording = not self.recording
                    state = "STARTED" if self.recording else "PAUSED"
                    print(f"[Recording {state}] Frames: {self.record_count}")
                if event.key == pygame.K_q:
                    quit_flag = True

        return quit_flag

    def save_frame(self, image: np.ndarray):
        filename = f"frame_{self.record_count:06d}.jpg"
        img_path = SAVE_DIR / "images" / filename
        cv2.imwrite(str(img_path), image)

        # KEY CHANGE: save BOTH direction and speed
        self.log.append({
            "image_path": str(img_path),
            "direction":  round(float(self.direction), 4),
            "speed":      round(float(self.speed),     4),
            "timestamp":  time.time()
        })
        self.record_count += 1

    def draw_hud(self):
        self.screen.fill((20, 20, 40))
        rec_text  = "● REC" if self.recording else "○ PAUSED"
        rec_color = (255, 80, 80) if self.recording else (160, 160, 160)
        progress  = min(self.record_count / TARGET_IMAGES * 100, 100)

        # Speed bar (visual)
        speed_bar_w = int(self.speed / THROTTLE_MAX * 200)
        dir_bar_x   = 250 + int(self.direction * 100)

        lines = [
            (f"{rec_text}  |  {self.record_count}/{TARGET_IMAGES}", rec_color),
            (f"Progress: [{('█' * int(progress // 5)).ljust(20)}] {progress:.0f}%",
             (100, 200, 255)),
            (f"Direction: {self.direction:+.2f}   Speed: {self.speed:.2f}",
             (200, 255, 200)),
            (f"W=go  S=brake  A/D=steer  R=record  Q=quit",
             (160, 160, 160)),
        ]
        for i, (text, color) in enumerate(lines):
            surf = self.font.render(text, True, color)
            self.screen.blit(surf, (20, 15 + i * 45))

        # Speed bar
        pygame.draw.rect(self.screen, (40, 40, 60), (20, 185, 200, 12))
        pygame.draw.rect(self.screen, (0, 200, 100), (20, 185, speed_bar_w, 12))

        # Direction indicator
        pygame.draw.rect(self.screen, (40, 40, 60), (200, 185, 200, 12))
        pygame.draw.rect(self.screen, (0, 150, 255), (dir_bar_x - 4, 183, 8, 16))

        pygame.display.flip()

    def save_manifest(self):
        manifest = SAVE_DIR / "labels.json"
        with open(manifest, "w") as f:
            json.dump(self.log, f, indent=2)
        print(f"\n[Saved] {self.record_count} frames → {SAVE_DIR}")
        print(f"[Saved] Manifest → {manifest}")

        # Print speed distribution summary
        speeds = [r["speed"] for r in self.log]
        dirs   = [r["direction"] for r in self.log]
        print(f"\n[Stats] Speed  — mean: {np.mean(speeds):.2f}  "
              f"min: {min(speeds):.2f}  max: {max(speeds):.2f}")
        print(f"[Stats] Direction — mean: {np.mean(np.abs(dirs)):.2f}  "
              f"turning%: {sum(1 for d in dirs if abs(d)>0.05)/len(dirs)*100:.0f}%")

    def run(self):
        """Run with Donkey Car Simulator."""
        try:
            import gym
            import gym_donkeycar

            env = gym.make(
                "donkey-warehouse-v0",
                conf={"host": SIM_HOST, "port": SIM_PORT}
            )
            obs, _ = env.reset()
            print("[Simulator] Connected.")

            while True:
                self.frame_count += 1
                quit_flag = self.get_keyboard_input()
                if quit_flag:
                    break

                action = [self.direction, self.speed]
                obs, reward, done, truncated, info = env.step(action)

                # Only record when actually moving (speed > 0.05)
                if self.recording and self.speed > 0.05:
                    self.save_frame(obs)

                self.draw_hud()

                if done:
                    obs, _ = env.reset()

                self.clock.tick(30)

            env.close()

        except ImportError:
            print("[Fallback] Simulator unavailable. Using webcam.")
            self._run_webcam()

    def _run_webcam(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            quit_flag = self.get_keyboard_input()
            if quit_flag:
                break

            if self.recording and self.speed > 0.05:
                self.save_frame(frame)

            self.draw_hud()
            self.clock.tick(30)

        cap.release()


def main():
    collector = DataCollectorV2()
    try:
        collector.run()
    finally:
        collector.save_manifest()
        pygame.quit()
        if collector.record_count < TARGET_IMAGES:
            print(f"\n[Warning] Only {collector.record_count} frames. "
                  f"Need {TARGET_IMAGES}. Do more laps.")
        else:
            print(f"\n[✓] {collector.record_count} frames collected. "
                  f"Ready to train!")


if __name__ == "__main__":
    main()