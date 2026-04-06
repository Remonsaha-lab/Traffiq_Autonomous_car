"""
============================================================
 TRAFFIQ v2 — Raspberry Pi Inference Script
 File: inference/run_car.py

 This is the script that runs ON THE CAR during the
 competition. It:
   1. Captures 640×480 frames from the Pi Camera
   2. Runs the OpenCV pipeline (lane + obstacle detection)
   3. Runs the TFLite CNN (speed + direction prediction)
   4. Merges both through the Decision Layer
   5. Sends [speed, direction] to the vehicle controller
   6. Safe-stops if anything goes wrong

 SAFE STOP MECHANISM (required by rules):
   A watchdog thread monitors the main loop.
   If the main loop freezes for >500ms, the watchdog
   calls stop_vehicle() independently.

 Usage (on Pi):
   python3 inference/run_car.py --model models/traffiq_v2.tflite
============================================================
"""

import sys
import time
import threading
import argparse
import numpy as np
import cv2
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from cv.cv_pipeline       import run_pipeline
from inference.decision_layer import DecisionLayer

# ── VEHICLE CONTROL INTERFACE ─────────────────────────────
# Replace these stubs with your actual motor/servo control
# once you have the physical car from organizers.

def set_speed_direction(speed: float, direction: float):
    """
    Send [speed, direction] to the vehicle.
    Both values are in [-1, 1].
      speed:     -1 = full reverse, 0 = stop, 1 = full forward
      direction: -1 = full left,   0 = straight, 1 = full right

    REPLACE THIS with actual GPIO/PWM calls for your car.
    """
    # Example GPIO stub:
    # gpio.set_pwm(MOTOR_PIN,    speed_to_pwm(speed))
    # gpio.set_pwm(STEERING_PIN, dir_to_pwm(direction))
    print(f"\r[CMD] speed={speed:+.3f}  direction={direction:+.3f}  ", end='', flush=True)


def stop_vehicle():
    """Emergency stop — called by watchdog if main loop hangs."""
    set_speed_direction(0.0, 0.0)
    print("\n[SAFE STOP] Vehicle halted.")


# ── WATCHDOG ──────────────────────────────────────────────

class Watchdog:
    """
    Monitors the main inference loop.
    If the loop doesn't check in within timeout_s seconds,
    it triggers an emergency stop.

    Required by TRAFFIQ rules:
    "The AI system must include a safe-stop mechanism
    that halts the vehicle in case of model crash,
    unexpected behavior, or loss of visual input."
    """
    def __init__(self, timeout_s: float = 0.5):
        self.timeout  = timeout_s
        self.last_ping = time.time()
        self.active   = True
        self._thread  = threading.Thread(
            target=self._watch, daemon=True
        )
        self._thread.start()

    def ping(self):
        """Call this every frame to tell watchdog you're alive."""
        self.last_ping = time.time()

    def stop(self):
        self.active = False

    def _watch(self):
        while self.active:
            time.sleep(0.1)
            if time.time() - self.last_ping > self.timeout:
                print(f"\n[WATCHDOG] No ping for {self.timeout}s — stopping!")
                stop_vehicle()
                self.active = False


# ── TFLite MODEL LOADER ───────────────────────────────────

class TFLiteModel:
    """
    Loads and runs the INT8 quantized TFLite model.
    Handles the INT8 ↔ float scaling automatically.
    """
    def __init__(self, model_path: str):
        try:
            import tflite_runtime.interpreter as tflite
            self.interpreter = tflite.Interpreter(model_path=model_path)
        except ImportError:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=model_path)

        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Read quantization parameters for INT8 ↔ float conversion
        inp_quant = self.input_details[0].get('quantization', (1.0, 0))
        out_quant = self.output_details[0].get('quantization', (1.0, 0))
        self.inp_scale, self.inp_zp = inp_quant
        self.out_scale, self.out_zp = out_quant

        print(f"[TFLite] Loaded: {model_path}")
        print(f"[TFLite] Input  shape: {self.input_details[0]['shape']}")
        print(f"[TFLite] Output shape: {self.output_details[0]['shape']}")

    def predict(self, image_float32: np.ndarray) -> np.ndarray:
        """
        Run inference on a single preprocessed image.

        Input:  float32 array shape (66, 200, 3) in [0, 1]
        Output: float32 array [speed, direction] in [-1, 1]
        """
        inp = image_float32[np.newaxis, ...]   # add batch dim → (1, 66, 200, 3)

        # Convert float32 → INT8 using quantization params
        if self.input_details[0]['dtype'] == np.int8:
            inp = (inp / self.inp_scale + self.inp_zp).astype(np.int8)

        self.interpreter.set_tensor(self.input_details[0]['index'], inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Convert INT8 → float32
        if self.output_details[0]['dtype'] == np.int8:
            out = (out.astype(np.float32) - self.out_zp) * self.out_scale

        return out[0]   # shape (2,) → [speed, direction]


# ── CAMERA SETUP ──────────────────────────────────────────

def init_camera():
    """
    Initialize PiCamera2 at 640×480.
    Falls back to OpenCV VideoCapture for testing on PC.
    """
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        config = cam.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        cam.configure(config)
        cam.start()
        time.sleep(0.5)   # warm-up

        def get_frame():
            return cam.capture_array()   # returns RGB directly

        print("[Camera] PiCamera2 initialized at 640×480")
        return get_frame, lambda: cam.stop()

    except ImportError:
        print("[Camera] picamera2 not found — using webcam (test mode)")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        def get_frame():
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Camera read failed")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return get_frame, cap.release


# ── MAIN LOOP ─────────────────────────────────────────────

def run(model_path: str, show_debug: bool = False):
    """
    Main inference loop. Runs until KeyboardInterrupt.
    """
    print("\n" + "="*50)
    print("  TRAFFIQ v2 — Starting Run")
    print("="*50 + "\n")

    # Load everything
    model    = TFLiteModel(model_path)
    decision = DecisionLayer()
    decision.reset()

    get_frame, release_camera = init_camera()
    watchdog = Watchdog(timeout_s=0.5)

    frame_count = 0
    fps_timer   = time.time()

    try:
        print("[Run] Loop starting. Press Ctrl+C to stop.\n")

        while True:
            t_start = time.time()

            # ── 1. Capture ───────────────────────────────
            try:
                raw_frame = get_frame()
            except RuntimeError as e:
                print(f"\n[ERROR] Camera failed: {e}")
                break

            # ── 2. OpenCV pipeline ───────────────────────
            pipeline_result = run_pipeline(raw_frame)
            cnn_input = pipeline_result['cnn_input']
            line      = pipeline_result['line']
            obstacles = pipeline_result['obstacles']

            # ── 3. CNN inference ─────────────────────────
            prediction    = model.predict(cnn_input)
            cnn_speed     = float(prediction[0])
            cnn_direction = float(prediction[1])

            # ── 4. Decision layer ────────────────────────
            final_speed, final_direction = decision.decide(
                cnn_speed, cnn_direction, line, obstacles
            )

            # ── 5. Send to vehicle ───────────────────────
            set_speed_direction(final_speed, final_direction)

            # ── 6. Ping watchdog ─────────────────────────
            watchdog.ping()

            # ── 7. FPS tracking ──────────────────────────
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_timer
                fps = 30 / elapsed
                fps_timer = time.time()
                print(
                    f"\n[FPS: {fps:.1f}] "
                    f"Line: {'YES' if line['found'] else ' NO'} "
                    f"Obs: {obstacles['count']} "
                    f"CNN→[{cnn_speed:+.2f},{cnn_direction:+.2f}] "
                    f"Final→[{final_speed:+.2f},{final_direction:+.2f}]"
                )

            # ── 8. Debug window (PC testing only) ────────
            if show_debug:
                debug_bgr = cv2.cvtColor(
                    pipeline_result['debug_frame'], cv2.COLOR_RGB2BGR
                )
                # Overlay final output on debug frame
                cv2.putText(
                    debug_bgr,
                    f"Speed:{final_speed:+.2f}  Dir:{final_direction:+.2f}",
                    (10, debug_bgr.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
                )
                cv2.imshow("TRAFFIQ v2 Debug", debug_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n[Stopped] Keyboard interrupt received.")

    finally:
        # Always stop the car before exiting
        stop_vehicle()
        watchdog.stop()
        release_camera()
        if show_debug:
            cv2.destroyAllWindows()

        print(f"[Done] Ran {frame_count} frames.")


# ── ENTRY POINT ───────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRAFFIQ v2 — Car Inference")
    parser.add_argument(
        "--model", type=str,
        default="models/traffiq_v2.tflite",
        help="Path to TFLite model file"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Show OpenCV debug window (PC testing only)"
    )
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"[ERROR] Model file not found: {args.model}")
        print("        Train the model first with train_v2.py")
        sys.exit(1)

    run(args.model, show_debug=args.debug)