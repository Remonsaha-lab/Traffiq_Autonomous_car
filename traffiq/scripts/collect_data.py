"""
============================================================
 TRAFFIQ — Simulator Data Collection Script
 Connects to the Donkey Car Simulator and records
 (image, steering, throttle) pairs for training.

 Controls:
   W / Up Arrow    → Throttle forward
   S / Down Arrow  → Brake / Reverse
   A / Left Arrow  → Steer left
   D / Right Arrow → Steer right
   R               → Start/Stop recording
   Q               → Quit and save dataset
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
SIM_HOST        = "127.0.0.1"
SIM_PORT        = 9091 # Port must be 9091 according to old simulator
IMG_WIDTH       = 160
IMG_HEIGHT      = 120
TARGET_IMAGES   = 5000          # Minimum for Round 1
SAVE_DIR        = Path("dataset") / datetime.now().strftime("%Y%m%d_%H%M%S")
STEERING_SCALE  = 1.0           # Adjust sensitivity
THROTTLE_DEFAULT = 0.3
# ──────────────────────────────────────────────────────────

class DataCollector:
    def __init__(self):
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        (SAVE_DIR / "images").mkdir(exist_ok=True)

        self.frame_count  = 0
        self.record_count = 0
        self.recording    = False
        self.steering     = 0.0
        self.throttle     = 0.0
        self.log          = []

        # Pygame for keyboard control
        pygame.init()
        self.screen = pygame.display.set_mode((400, 200))
        pygame.display.set_caption("TRAFFIQ — Data Collector")
        self.font = pygame.font.SysFont("monospace", 18)
        self.clock = pygame.time.Clock()

        print(f"\n[DataCollector] Save directory: {SAVE_DIR}")
        print("[DataCollector] Press R to start recording, Q to quit.\n")

    def get_keyboard_input(self):
        """Read keyboard and return (steering, throttle, recording, quit)"""
        quit_flag = False
        keys = pygame.key.get_pressed()

        # Steering
        if keys[pygame.K_LEFT]  or keys[pygame.K_a]: self.steering = -STEERING_SCALE
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: self.steering =  STEERING_SCALE
        else: self.steering = 0.0

        # Throttle
        if keys[pygame.K_UP]   or keys[pygame.K_w]: self.throttle = THROTTLE_DEFAULT
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: self.throttle = -0.2
        else: self.throttle = 0.0

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_flag = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.recording = not self.recording
                    state = "STARTED" if self.recording else "PAUSED"
                    print(f"[Recording {state}] Frames so far: {self.record_count}")
                if event.key == pygame.K_q:
                    quit_flag = True

        return quit_flag

    def save_frame(self, image: np.ndarray):
        """Save image + label to disk"""
        filename = f"frame_{self.record_count:06d}.jpg"
        img_path = SAVE_DIR / "images" / filename
        cv2.imwrite(str(img_path), image)

        self.log.append({
            "image_path": str(img_path),
            "steering":   round(float(self.steering), 4),
            "throttle":   round(float(self.throttle), 4),
            "timestamp":  time.time()
        })
        self.record_count += 1

    def draw_hud(self, image: np.ndarray):
        """Overlay HUD info on the pygame window"""
        self.screen.fill((20, 20, 40))

        # Status
        rec_text  = "● REC" if self.recording else "○ PAUSED"
        rec_color = (255, 80, 80) if self.recording else (160, 160, 160)
        progress  = min(self.record_count / TARGET_IMAGES * 100, 100)

        lines = [
            (f"{rec_text}  |  Frames: {self.record_count}/{TARGET_IMAGES}", rec_color),
            (f"Progress: [{('█' * int(progress//5)).ljust(20)}] {progress:.1f}%", (100, 200, 255)),
            (f"Steering: {self.steering:+.2f}   Throttle: {self.throttle:.2f}", (200, 255, 200)),
            (f"Controls: W/S=throttle  A/D=steer  R=rec  Q=quit", (180, 180, 180)),
        ]
        for i, (text, color) in enumerate(lines):
            surf = self.font.render(text, True, color)
            self.screen.blit(surf, (20, 20 + i * 40))

        pygame.display.flip()

    def save_dataset_manifest(self):
        """Write all labels to a single JSON file"""
        manifest_path = SAVE_DIR / "labels.json"
        with open(manifest_path, "w") as f:
            json.dump(self.log, f, indent=2)
        print(f"\n[Saved] {self.record_count} frames → {SAVE_DIR}")
        print(f"[Saved] Manifest → {manifest_path}")

    def run_with_sim(self):
        """Main loop — connects to Donkey Car Sim via gym-donkeycar"""
        try:
            import gym
            import gym_donkeycar

            # In some older versions we actually need to import the module that registers the envs explicitly
            import gym_donkeycar.envs.donkey_env

            conf = {
                "host": SIM_HOST,
                "port": SIM_PORT,
            }

            print(f"[Simulator] Connecting to simulator at {SIM_HOST}:{SIM_PORT}...")
            print("[Simulator] Make sure the Donkey Car Simulator is already running!")
            # The older gym-donkeycar environment name is usually "donkey-warehouse-v0"
            # It must be registered before `gym.make` works
            # In old gym-donkeycar versions, importing gym_donkeycar.envs registers them.
            import gym_donkeycar.envs
            os.environ["DONKEY_SIM_PATH"] = os.path.expanduser("~/donkey_sim/DonkeySimLinux/donkey_sim.x86_64")
            os.environ["DONKEY_SIM_PORT"] = str(SIM_PORT)
            os.environ["DONKEY_SIM_HEADLESS"] = "0"
            env = gym.make("donkey-warehouse-v0")
            obs = env.reset()  # gym<0.26 API returns just obs
            info = {}
            print("[Simulator] Connected to Donkey Car Simulator.")

            while True:
                self.frame_count += 1
                quit_flag = self.get_keyboard_input()
                if quit_flag:
                    break

                # Step the simulation
                action = np.array([self.steering, self.throttle], dtype=np.float32)
                
                # gym API: step() returns (obs, reward, done, info)
                result = env.step(action)
                if len(result) == 4:
                    obs, reward, terminated, info = result
                    truncated = False
                else:
                    # Fallback for newer Gym API giving 5 values
                    obs, reward, terminated, truncated, info = result

                # obs is the camera image (120x160x3)
                image = obs

                if self.recording and self.throttle > 0:
                    self.save_frame(image)

                self.draw_hud(image)

                if terminated or truncated:
                    obs = env.reset()

                self.clock.tick(30)  # 30 FPS cap

            env.close()

        except ImportError:
            print("[WARNING] gym-donkeycar not found. Running in MOCK mode.")
            self.run_mock_mode()
        except Exception as e:
            print(f"[WARNING] Could not connect to simulator: {e}")
            print("[WARNING] Make sure the Donkey Car Simulator is running on port", SIM_PORT)
            print("[WARNING] Falling back to MOCK mode (webcam).")
            self.run_mock_mode()

    def run_mock_mode(self):
        """
        Mock mode — use your webcam as a stand-in if sim isn't available.
        Useful for testing the data collection pipeline itself.
        """
        print("[Mock Mode] Using webcam. Ensure camera is connected.")
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
            quit_flag = self.get_keyboard_input()
            if quit_flag:
                break

            if self.recording and self.throttle > 0:
                self.save_frame(frame)

            self.draw_hud(frame)
            self.clock.tick(30)

        cap.release()


def main():
    collector = DataCollector()
    try:
        collector.run_with_sim()
    finally:
        collector.save_dataset_manifest()
        pygame.quit()
        print("\n[Done] Data collection complete.")
        if collector.record_count < TARGET_IMAGES:
            print(f"[Warning] Only {collector.record_count} frames collected.")
            print(f"          Target is {TARGET_IMAGES}. Do more laps!")
        else:
            print(f"[✓] Target of {TARGET_IMAGES} frames reached. Ready to train!")


if __name__ == "__main__":
    main()