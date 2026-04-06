"""
============================================================
 TRAFFIQ v2 — OpenCV Vision Pipeline
 File: cv/cv_pipeline.py

 This file handles ALL computer vision processing BEFORE
 the neural network sees the image. It does three jobs:

   1. WHITE LINE DETECTION
      Finds the white center line on the black track.
      Computes how far the car is from the line center.
      This gives a raw direction signal.

   2. OBSTACLE DETECTION
      Finds objects placed on the black surface.
      Obstacles are anything significantly brighter/colored
      than the surrounding black track.
      Computes their position (left/right) and proximity.

   3. LIGHTING NORMALIZATION
      Compensates for variations in arena lighting color
      so that white line and obstacle detection stay
      reliable regardless of lighting conditions.

 OUTPUT of this pipeline (used by both training and
 inference):
   - A preprocessed image fed into the CNN
   - A feature dictionary used by the decision layer
============================================================
"""

import cv2
import numpy as np


# ─── CONFIGURATION ────────────────────────────────────────

# Input resolution (640×480 confirmed minimum)
INPUT_W = 640
INPUT_H = 480

# CNN input size (DAVE-2 standard)
CNN_W = 200
CNN_H = 66

# Crop bounds (fraction of INPUT_H)
# Top 25% = ceiling/sky, bottom 15% = car hood
CROP_TOP    = int(INPUT_H * 0.25)   # 120px
CROP_BOTTOM = int(INPUT_H * 0.85)   # 408px

# White line detection — HSV thresholds for "white"
# We use HSV because white = high V, low S regardless of lighting hue
# WHITE_S_MAX = 60    # low saturation  → white / near-white
# WHITE_V_MIN = 180   # high brightness → bright
# ADDED: Relaxed thresholds to work with real-world/darker images
WHITE_S_MAX = 100
WHITE_V_MIN = 110

# Obstacle detection — minimum contour area (pixels²)
# Anything smaller is noise, anything larger is an obstacle
OBSTACLE_MIN_AREA = 800
OBSTACLE_MAX_AREA = 80000

# Region of Interest for line detection:
# Only look at bottom half of the cropped image
# (line markings are on the ground directly ahead)
LINE_ROI_START = 0.5   # 50% down the cropped image


# ─── STEP 1: LIGHTING NORMALIZATION ───────────────────────

def normalize_lighting(image: np.ndarray) -> np.ndarray:
    """
    Compensates for variations in arena lighting color.

    The arena may have:
      - Yellow/warm overhead lights
      - Cool/blue LED strips
      - Shadows from walls or obstacles
      - Bright spots under direct lighting

    Approach: CLAHE (Contrast Limited Adaptive Histogram
    Equalization) on the L channel of LAB color space.

    WHY LAB?
      LAB separates lightness (L) from color (A=green-red,
      B=blue-yellow). We equalize only the lightness channel,
      leaving color information untouched. This enhances
      contrast uniformly without distorting colors —
      important because we use color to detect obstacles.

    WHY CLAHE over plain histogram equalization?
      Plain HE applies one global contrast curve to the whole
      image. CLAHE divides the image into small tiles and
      equalizes each tile independently, then blends them.
      Result: dark corners get brightened while already-bright
      areas don't get washed out. Better for scenes with
      both shadows and bright spots simultaneously.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Create CLAHE object:
    # clipLimit=2.0 prevents over-amplifying noise in flat regions
    # tileGridSize=(8,8) means 8×8 grid of tiles over the image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Apply only to L channel (lightness)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ─── STEP 2: CROP ─────────────────────────────────────────

def crop_frame(image: np.ndarray) -> np.ndarray:
    """
    Remove sky/ceiling (top) and car hood (bottom).

    Input:  640×480 full frame
    Output: 640×288 cropped frame (120px removed top,
                                    72px removed bottom)

    WHY CROP?
      The CNN only needs to see the track surface.
      Sky and hood pixels waste network capacity and can
      introduce spurious patterns (e.g. the hood's edge
      could be mistaken for a line).
    """
    # return image[CROP_TOP:CROP_BOTTOM, :, :]
    
    # ADDED: Dynamically crop to prevent empty image errors with different sizes
    h = image.shape[0]
    c_top = int(h * 0.25)
    c_bot = int(h * 0.85)
    return image[c_top:c_bot, :, :]


# ─── STEP 3: WHITE LINE DETECTION ─────────────────────────

def detect_white_line(cropped: np.ndarray) -> dict:
    """
    Finds the white center line on the black track.

    Returns a dict:
      'found'     : bool   — was a line detected?
      'cx_norm'   : float  — normalized center x of line
                             0.0 = far left, 1.0 = far right
                             0.5 = perfectly centered
      'offset'    : float  — deviation from center [-1, 1]
                             negative = line is to the left
                             positive = line is to the right
      'area'      : int    — pixel area of detected line
      'debug_mask': ndarray — binary mask for visualization

    HOW IT WORKS:
      1. Convert to HSV
      2. Threshold for white pixels (high V, low S)
      3. Apply morphological operations to clean the mask
      4. Find contours in the lower ROI of the image
      5. Largest contour = the center line
      6. Compute its centroid and convert to offset
    """
    h, w = cropped.shape[:2]

    # Convert to HSV for color-invariant white detection
    hsv = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)

    # Threshold: white = any hue, low saturation, high value
    # np.array([0, 0, WHITE_V_MIN])   → lower bound
    # np.array([180, WHITE_S_MAX, 255]) → upper bound
    white_mask = cv2.inRange(
        hsv,
        np.array([0,   0,           WHITE_V_MIN]),
        np.array([180, WHITE_S_MAX, 255        ])
    )

    # Morphological operations to clean the mask:
    # OPEN  = erode then dilate → removes small noise specks
    # CLOSE = dilate then erode → fills gaps in the line
    kernel = np.ones((5, 5), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN,  kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    # Only look at the bottom half of the cropped image
    # (the line directly ahead of the car, not in the distance)
    roi_start = int(h * LINE_ROI_START)
    roi_mask  = np.zeros_like(white_mask)
    roi_mask[roi_start:, :] = white_mask[roi_start:, :]

    # Find contours in the ROI
    contours, _ = cv2.findContours(
        roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return {
            'found': False, 'cx_norm': 0.5,
            'offset': 0.0,  'area': 0,
            'debug_mask': roi_mask
        }

    # Take the largest contour — most likely the center line
    largest = max(contours, key=cv2.contourArea)
    area    = cv2.contourArea(largest)

    if area < 200:   # too small — probably noise
        return {
            'found': False, 'cx_norm': 0.5,
            'offset': 0.0,  'area': int(area),
            'debug_mask': roi_mask
        }

    # Compute centroid of the contour using image moments
    # M['m10']/M['m00'] = x centroid
    # M['m01']/M['m00'] = y centroid
    M  = cv2.moments(largest)
    cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else w // 2

    # Normalize cx to [0, 1] range
    cx_norm = cx / w

    # Compute offset: how far is the line from image center?
    # Image center = 0.5 in normalized coords
    # offset > 0 → line is right of center → steer right
    # offset < 0 → line is left of center  → steer left
    offset = (cx_norm - 0.5) * 2   # maps [0,1] → [-1, 1]

    return {
        'found':      True,
        'cx_norm':    round(cx_norm, 4),
        'offset':     round(float(offset), 4),
        'area':       int(area),
        'debug_mask': roi_mask
    }


# ─── STEP 4: OBSTACLE DETECTION ───────────────────────────

def detect_obstacles(cropped: np.ndarray) -> dict:
    """
    Finds obstacles placed on the black track surface.

    The track is BLACK. Obstacles are physically placed
    objects — they will be visually distinct: brighter,
    colored, or textured compared to the surrounding surface.

    Returns a dict:
      'found'       : bool  — any obstacle detected?
      'count'       : int   — number of obstacles found
      'nearest'     : dict  — closest/largest obstacle info
          'cx_norm' : float — normalized x position [0,1]
          'cy_norm' : float — normalized y position [0,1]
                              (1.0 = bottom = very close)
          'area_norm': float — size as fraction of image
          'side'    : str   — 'left', 'center', or 'right'
      'all_obstacles': list — all detected obstacles
      'debug_mask'  : ndarray — binary mask

    HOW IT WORKS:
      1. Convert to grayscale
      2. Threshold: black surface is near 0, obstacles > 30
      3. Remove the white line from the obstacle mask
         (white line would otherwise be detected as obstacle)
      4. Find contours within reasonable size range
      5. For each contour compute center and size
      6. Classify as left/center/right based on x position
    """
    h, w = cropped.shape[:2]

    # Convert to grayscale for brightness-based detection
    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    
    # ADDED: Blur the ground to reduce texture/noise before detection
    gray_blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Adaptive threshold: handles uneven lighting better than
    # a fixed threshold. blockSize=51 means each 51×51 pixel
    # neighbourhood gets its own threshold value.
    # C=10 subtracts a constant from the mean — helps separate
    # objects from background when they are similar brightness.
    # ADDED: C=-20 instead of 10. By subtracting a negative (adding 20),
    # we ensure only objects SIGNIFICANTLY brighter than the local average
    # are kept. A positive C flags the entire background!
    # obstacle_mask = cv2.adaptiveThreshold(
    #     gray, 255,
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY,
    #     blockSize=51, C=10
    # )
    obstacle_mask = cv2.adaptiveThreshold(
        gray_blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=51, C=-20
    )

    # Remove white line pixels from obstacle mask.
    # Get white mask and subtract it.
    hsv        = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
    white_mask = cv2.inRange(
        hsv,
        np.array([0,   0,           WHITE_V_MIN]),
        np.array([180, WHITE_S_MAX, 255        ])
    )
    # Dilate white mask slightly before subtracting
    # to fully erase the line's edges
    white_dilated = cv2.dilate(
        white_mask, np.ones((7, 7), np.uint8), iterations=1
    )
    obstacle_mask = cv2.bitwise_and(
        obstacle_mask,
        cv2.bitwise_not(white_dilated)
    )

    # Clean up with morphological opening
    kernel = np.ones((7, 7), np.uint8)
    obstacle_mask = cv2.morphologyEx(
        obstacle_mask, cv2.MORPH_OPEN, kernel
    )

    # Find contours
    contours, _ = cv2.findContours(
        obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    obstacles = []
    total_area = h * w

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if not (OBSTACLE_MIN_AREA < area < OBSTACLE_MAX_AREA):
            continue

        M   = cv2.moments(cnt)
        if M['m00'] == 0:
            continue

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        cx_norm   = cx / w
        cy_norm   = cy / h       # 1.0 = bottom = very close
        area_norm = area / total_area

        # Classify horizontal position
        if cx_norm < 0.35:
            side = 'left'
        elif cx_norm > 0.65:
            side = 'right'
        else:
            side = 'center'

        obstacles.append({
            'cx_norm':   round(cx_norm,   4),
            'cy_norm':   round(cy_norm,   4),
            'area_norm': round(area_norm, 4),
            'side':      side,
            'area_px':   int(area)
        })

    if not obstacles:
        return {
            'found':         False,
            'count':         0,
            'nearest':       None,
            'all_obstacles': [],
            'debug_mask':    obstacle_mask
        }

    # "Nearest" = largest area_norm (close obstacles appear bigger)
    nearest = max(obstacles, key=lambda o: o['area_norm'])

    return {
        'found':         True,
        'count':         len(obstacles),
        'nearest':       nearest,
        'all_obstacles': obstacles,
        'debug_mask':    obstacle_mask
    }


# ─── STEP 5: CNN PREPROCESSING ────────────────────────────

def preprocess_for_cnn(cropped: np.ndarray) -> np.ndarray:
    """
    Converts the cropped 640×288 RGB image into the
    66×200 YUV tensor the CNN expects.

    Steps:
      1. Resize to 200×66
      2. Convert RGB → YUV

    WHY YUV?
      Y channel = brightness = where lane lines are visible
      U,V channels = color = helps distinguish obstacles
      Separating brightness from color gives the CNN
      cleaner signals for both tasks simultaneously.
    """
    img = cv2.resize(cropped, (CNN_W, CNN_H))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    return img.astype(np.float32) / 255.0


# ─── MAIN PIPELINE FUNCTION ───────────────────────────────

def run_pipeline(raw_frame: np.ndarray) -> dict:
    """
    Master function. Takes a raw 640×480 RGB frame and
    returns everything the system needs.

    Returns:
      'cnn_input'   : np.ndarray (66, 200, 3) float32
                      Ready to feed into the neural network.

      'line'        : dict from detect_white_line()
      'obstacles'   : dict from detect_obstacles()

      'debug_frame' : np.ndarray — annotated frame for
                      visualization during development
    """
    # Step 1: normalize lighting
    normalized = normalize_lighting(raw_frame)

    # Step 2: crop
    cropped = crop_frame(normalized)

    # Step 3: detect line
    line = detect_white_line(cropped)

    # Step 4: detect obstacles
    obstacles = detect_obstacles(cropped)

    # Step 5: prepare CNN input
    cnn_input = preprocess_for_cnn(cropped)

    # Step 6: build annotated debug frame
    debug = draw_debug(cropped.copy(), line, obstacles)

    return {
        'cnn_input':   cnn_input,
        'line':        line,
        'obstacles':   obstacles,
        'debug_frame': debug
    }


# ─── DEBUG VISUALIZATION ──────────────────────────────────

def draw_debug(frame: np.ndarray, line: dict, obs: dict) -> np.ndarray:
    """
    Draws detection results onto the frame for development.
    This is NOT used during the actual car run.
    """
    h, w = frame.shape[:2]

    # Draw image center line (target)
    cv2.line(frame, (w//2, 0), (w//2, h), (100, 100, 100), 1)

    # Draw white line detection result
    if line['found']:
        cx_px = int(line['cx_norm'] * w)
        cv2.line(frame, (cx_px, 0), (cx_px, h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Line offset: {line['offset']:+.2f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2
        )
    else:
        cv2.putText(
            frame, "Line: NOT FOUND",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 255), 2
        )

    # Draw obstacle detections
    if obs['found']:
        for o in obs['all_obstacles']:
            cx_px = int(o['cx_norm'] * w)
            cy_px = int(o['cy_norm'] * h)
            radius = int(np.sqrt(o['area_px'] / np.pi))
            cv2.circle(frame, (cx_px, cy_px), radius, (255, 80, 0), 2)
            cv2.putText(
                frame,
                f"{o['side']}",
                (cx_px - 20, cy_px - radius - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 0), 1
            )
        cv2.putText(
            frame,
            f"Obstacles: {obs['count']}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 80, 0), 2
        )

    return frame


# ─── STANDALONE TEST ──────────────────────────────────────

if __name__ == "__main__":
    """
    Test the pipeline on your webcam or a test image.
    Run: python3 cv/cv_pipeline.py
    Press Q to quit.
    """
    print("[CV Pipeline Test] Starting webcam...")
    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH,  INPUT_W)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_H)
    # given you image
    image_path = "/home/remon/Documents/Trafiic_car_autonomous/traffiq/scripts/images.jpeg"
    frame = cv2.imread(image_path)
    

    while True:
        # ret, frame = cap.read()
        # if not ret:
        #     break

        # ADDED: Resize image to expected minimum width and height to prevent processing errors
        frame_resized = cv2.resize(frame, (INPUT_W, INPUT_H))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result    = run_pipeline(frame_rgb)

        debug_bgr = cv2.cvtColor(result['debug_frame'], cv2.COLOR_RGB2BGR)
        cv2.imshow("TRAFFIQ CV Pipeline", debug_bgr)

        # Print results
        line = result['line']
        obs  = result['obstacles']
        print(
            f"\rLine: {'YES' if line['found'] else ' NO '} "
            f"offset={line['offset']:+.3f} | "
            f"Obstacles: {obs['count']}",
            end='', flush=True
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[Done]")