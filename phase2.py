
import depthai as dai
import numpy as np
import cv2
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import mediapipe as mp
import time
import threading
import os
import tempfile

# ============== TEXT-TO-SPEECH SETUP (Google TTS) ========
from gtts import gTTS
import pygame

pygame.mixer.init()
tts_lock = threading.Lock()

def speak(text):
    """Speak text using Google TTS in a separate thread (non-blocking)"""
    def _speak():
        with tts_lock:
            try:
                # Create temporary mp3 file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                    temp_path = fp.name
                
                # Generate speech
                tts = gTTS(text=text, lang='en')
                tts.save(temp_path)
                
                # Play audio
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                # Cleanup
                os.unlink(temp_path)
            except Exception as e:
                print(f"[TTS Error] {e}")
    
    thread = threading.Thread(target=_speak, daemon=True)
    thread.start()
# =========================================================

# ============== CAMERA & TRANSFORM SETTINGS ==============
CAMERA_HEIGHT = 580       # mm
TILT_DEG = 50             # degrees
CAMERA_TO_ROBOT_Y = 700   # mm
# =========================================================

# ============== YELLOW COLOR DETECTION (HSV) =============
YELLOW_HSV_LOWER = np.array([20, 100, 100])   # Hue 20-40 for yellow
YELLOW_HSV_UPPER = np.array([40, 255, 255])
YELLOW_MIN_AREA = 500     # Minimum contour area in pixels
# =========================================================

# ============== HANDOVER SETTINGS ========================
SAFE_DISTANCE = 100       # mm - maintain 10cm from object
HAND_STABILITY_TIME = 2.0 # seconds - hand must be stable
GRIPPER_WAIT_TIME = 1.0   # seconds - wait before closing gripper

# GIVE MODE - Effort/Current Sensing
GIVE_HAND_STABLE_TIME = 3.0   # seconds - hand must be stable before locking position
EFFORT_THRESHOLD = 5.0        # effort delta to detect human pull (tune this!)
EFFORT_STABLE_TIME = 0.2      # seconds - pull must be sustained
# =========================================================

# ============== ROBOT CUSTOM HOME POSITION ===============
CUSTOM_HOME_POSITION = {
    'waist': 0.000,
    'shoulder': -0.800,
    'elbow': 1.300,
    'wrist_angle': 0.500,
    'wrist_rotate': 0.000
}
# =========================================================

def build_transform(camera_height, tilt_deg, camera_to_robot_y):
    tilt = np.radians(tilt_deg)
    T = np.array([
        [1,   0,             0,              0               ],
        [0,  -np.sin(tilt),  np.cos(tilt),  -camera_to_robot_y],
        [0,  -np.cos(tilt), -np.sin(tilt),   camera_height   ],
        [0,   0,             0,              1               ]
    ])
    return T

def pixel_to_camera_frame(x_pixel, y_pixel, depth_mm, intrinsics):
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    x_cam = (x_pixel - cx) * depth_mm / fx
    y_cam = (y_pixel - cy) * depth_mm / fy
    z_cam = depth_mm
    
    return np.array([x_cam, y_cam, z_cam, 1])


class YellowObjectDetector:
    """Detect yellow objects using HSV color thresholding"""
    
    def __init__(self):
        self.lower = YELLOW_HSV_LOWER
        self.upper = YELLOW_HSV_UPPER
        self.min_area = YELLOW_MIN_AREA
        
    def detect(self, frame):
        """
        Detect yellow objects in frame using color thresholding.
        Returns list of detections with same format as YOLO version.
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for yellow color
        mask = cv2.inRange(hsv, self.lower, self.upper)
        
        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill holes
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > self.min_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate center
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Calculate confidence based on area (larger = more confident)
                # Normalize to 0-1 range (adjust max_area as needed)
                max_area = 50000
                confidence = min(area / max_area, 1.0)
                
                detections.append({
                    'bbox': (x, y, x + w, y + h),
                    'center': (center_x, center_y),
                    'label': 'yellow_object',
                    'confidence': confidence,
                    'area': area
                })
        
        # Sort by area (largest first) and return
        detections.sort(key=lambda d: d['area'], reverse=True)
        return detections
    
    def get_mask(self, frame):
        """Return the color mask for debugging"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask


class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        if results.multi_hand_landmarks:
            hand_lm = results.multi_hand_landmarks[0]
            h, w = frame.shape[:2]
            
            wrist = hand_lm.landmark[0]
            middle_base = hand_lm.landmark[9]
            
            cx = int((wrist.x + middle_base.x) / 2 * w)
            cy = int((wrist.y + middle_base.y) / 2 * h)
            
            return {'center': (cx, cy), 'landmarks': hand_lm}
        return None
    
    def draw(self, frame, hand_data):
        if hand_data:
            self.mp_draw.draw_landmarks(
                frame, hand_data['landmarks'], self.mp_hands.HAND_CONNECTIONS
            )
            cv2.circle(frame, hand_data['center'], 10, (0, 255, 255), -1)


def go_to_custom_home(bot):
    bot.arm.set_joint_positions([
        CUSTOM_HOME_POSITION['waist'],
        CUSTOM_HOME_POSITION['shoulder'],
        CUSTOM_HOME_POSITION['elbow'],
        CUSTOM_HOME_POSITION['wrist_angle'],
        CUSTOM_HOME_POSITION['wrist_rotate']
    ])


def is_reachable(pos, min_reach=50, max_reach=400):
    dist = np.sqrt(pos[0]**2 + pos[1]**2)
    return min_reach < dist < max_reach and -100 < pos[2] < 350


def get_gripper_effort(bot):
    """Read current gripper effort/torque value"""
    try:
        js = bot.gripper.core.joint_states
        if js and hasattr(js, 'effort') and len(js.effort) > 0:
            effort = js.effort[0] if hasattr(js.effort, '__getitem__') else js.effort
            return effort
    except:
        pass
    return None


def calculate_approach_position(object_pos, safe_distance):
    x, y, z = object_pos[0], object_pos[1], object_pos[2]
    dist_xy = np.sqrt(x**2 + y**2)
    
    if dist_xy < 1:
        return None
    
    ux = x / dist_xy
    uy = y / dist_xy
    
    target_x = x - ux * safe_distance
    target_y = y - uy * safe_distance
    target_z = z
    
    return np.array([target_x, target_y, target_z])


def main():
    T_robot_camera = build_transform(CAMERA_HEIGHT, TILT_DEG, CAMERA_TO_ROBOT_Y)
    print("=" * 60)
    print("PROACTIVE HANDOVER SYSTEM (Color Detection)")
    print("=" * 60)
    print("Transform matrix:")
    print(np.round(T_robot_camera, 3))
    print()

    # Initialize robot
    print("Initializing RX200...")
    bot = InterbotixManipulatorXS(
        robot_model='rx200',
        group_name='arm',
        gripper_name='gripper'
    )
    
    print("Moving to custom home position...")
    go_to_custom_home(bot)
    
    # START WITH GRIPPER CLOSED
    print("Closing gripper...")
    bot.gripper.set_pressure(1.0)
    time.sleep(0.5)
    bot.gripper.grasp()
    time.sleep(1.0)
    print("Gripper closed!")
    print("Robot ready!\n")

    # Load detectors
    print("Initializing yellow color detector...")
    yellow_detector = YellowObjectDetector()
    print(f"Yellow HSV range: {YELLOW_HSV_LOWER} - {YELLOW_HSV_UPPER}")
    print("Color detector ready!")
    
    print("Initializing hand detector...")
    hand_detector = HandDetector()
    print("Hand detector ready!")
    
    print("Initializing text-to-speech (Google TTS)...")
    print("TTS ready!\n")

    # Pipeline
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setInterleaved(False)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)

    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)

    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    xout_depth.setStreamName("depth")

    cam_rgb.preview.link(xout_rgb.input)
    stereo.depth.link(xout_depth.input)

    # States
    STATE_FOLLOW = 0
    STATE_WAIT_FOR_HAND = 1
    STATE_RETURN_HOME = 2      # Go home after receiving object
    STATE_WAIT_HAND_ENTER = 3  # Wait for hand to enter frame
    STATE_TRACK_HAND = 4
    STATE_WAIT_FOR_PULL = 5    # Position locked, waiting for human to pull
    STATE_GIVE = 6
    
    state = STATE_FOLLOW
    last_target = None
    hand_stable_start = None
    gripper_timer_start = None
    locked_position = None    # Store position when locking
    baseline_effort = None    # Baseline effort for pull detection
    pull_start = None         # Time when pull was first detected
    show_mask = False
    
    # Voice command flags (to avoid repeating)
    spoken_state = None
    spoken_hand_detected = False
    startup_spoken = False  # Toggle for debug mask view

    with dai.Device(pipeline) as device:
        calib = device.readCalibration()
        intr = calib.getCameraIntrinsics(dai.CameraBoardSocket.RGB, 640, 480)
        intrinsics = {
            'fx': intr[0][0], 'fy': intr[1][1],
            'cx': intr[0][2], 'cy': intr[1][2]
        }
        print(f"Camera intrinsics: fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}")
        print("\nControls:")
        print("  'q' - quit")
        print("  'r' - reset to home")
        print("  'o' - open gripper (test)")
        print("  'c' - close gripper (test)")
        print("  'm' - toggle mask view")
        print(f"\nEffort sensing: threshold={EFFORT_THRESHOLD}, stable_time={EFFORT_STABLE_TIME}s\n")

        q_rgb = device.getOutputQueue("rgb", 4, False)
        q_depth = device.getOutputQueue("depth", 4, False)

        try:
            while True:
                rgb_frame = q_rgb.get().getCvFrame()
                depth_frame = q_depth.get().getFrame()
                depth_resized = cv2.resize(depth_frame, (640, 480))

                display = rgb_frame.copy()
                
                # Startup voice (only once)
                if not startup_spoken:
                    speak("Beep boop... just kidding. I'm EchoHand, nice to meet you!")
                    startup_spoken = True

                # Detect yellow object using color detection
                yellow_detections = yellow_detector.detect(rgb_frame)
                object_pos = None
                
                if yellow_detections:
                    # Use the largest detection (first in sorted list)
                    det = yellow_detections[0]
                    x, y = det['center']
                    x1, y1, x2, y2 = det['bbox']
                    depth_mm = depth_resized[int(y), int(x)]

                    if depth_mm > 0:
                        P_cam = pixel_to_camera_frame(x, y, depth_mm, intrinsics)
                        P_robot = T_robot_camera @ P_cam
                        object_pos = P_robot[:3]

                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.circle(display, (x, y), 5, (0, 255, 255), -1)
                        
                        info = f"Yellow: X={object_pos[0]:.0f} Y={object_pos[1]:.0f} Z={object_pos[2]:.0f}"
                        cv2.putText(display, info, (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        # Show confidence/area
                        conf_text = f"Area: {det['area']:.0f}px"
                        cv2.putText(display, conf_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Detect hand
                hand_data = hand_detector.detect(rgb_frame)
                hand_pos = None
                hand_detected = False
                
                if hand_data:
                    hx, hy = hand_data['center']
                    hx = max(0, min(hx, 639))
                    hy = max(0, min(hy, 479))
                    depth_mm = depth_resized[int(hy), int(hx)]
                    
                    if depth_mm > 0:
                        P_cam = pixel_to_camera_frame(hx, hy, depth_mm, intrinsics)
                        P_robot = T_robot_camera @ P_cam
                        hand_pos = P_robot[:3]
                        hand_detected = True
                        hand_detector.draw(display, hand_data)

                # STATE MACHINE
                if state == STATE_FOLLOW:
                    # Voice for entering this state
                    if spoken_state != STATE_FOLLOW:
                        speak("Time to follow you, my master...")
                        spoken_state = STATE_FOLLOW
                        spoken_hand_detected = False
                    
                    cv2.putText(display, "STATE: FOLLOWING OBJECT", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    if object_pos is not None:
                        target = calculate_approach_position(object_pos, SAFE_DISTANCE)
                        
                        if target is not None and is_reachable(target):
                            if last_target is None or np.linalg.norm(target - last_target) > 10:
                                x_m = target[0] / 1000.0
                                y_m = target[1] / 1000.0
                                z_m = max(target[2] / 1000.0, 0.02)
                                
                                print(f"Moving to: X={x_m:.3f} Y={y_m:.3f} Z={z_m:.3f}")
                                bot.arm.set_ee_pose_components(x=x_m, y=y_m, z=z_m, moving_time=1.8)
                                last_target = target
                        
                            if hand_detected and hand_pos is not None:
                                hand_to_obj = np.linalg.norm(hand_pos - object_pos)
                                
                                if hand_to_obj < 150:
                                    if hand_stable_start is None:
                                        hand_stable_start = time.time()
                                    elif time.time() - hand_stable_start > HAND_STABILITY_TIME:
                                        # Voice: Hand detected, opening gripper
                                        if not spoken_hand_detected:
                                            speak("Oh, a hand! Opening up...")
                                            spoken_hand_detected = True
                                        print("Hand stable! Opening gripper...")
                                        bot.gripper.release(2.0)
                                        time.sleep(2.5)
                                        print("Gripper opened!")
                                        state = STATE_WAIT_FOR_HAND
                                        hand_stable_start = None
                                        gripper_timer_start = time.time()
                                else:
                                    hand_stable_start = None

                elif state == STATE_WAIT_FOR_HAND:
                    # Voice for entering this state
                    if spoken_state != STATE_WAIT_FOR_HAND:
                        speak("I'm ready, hand it over.")
                        spoken_state = STATE_WAIT_FOR_HAND
                    
                    elapsed = time.time() - gripper_timer_start
                    remaining = GRIPPER_WAIT_TIME - elapsed
                    
                    cv2.putText(display, f"STATE: GRIPPER OPEN - Place object ({remaining:.1f}s)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(display, f"Timer: {elapsed:.1f}/{GRIPPER_WAIT_TIME:.1f}s", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    print(f"Waiting... {elapsed:.1f}/{GRIPPER_WAIT_TIME:.1f}s", end='\r')
                    
                    if elapsed >= GRIPPER_WAIT_TIME:
                        print(f"\nTimer complete! Closing gripper...")
                        bot.gripper.set_pressure(1.0)
                        time.sleep(0.5)
                        bot.gripper.grasp()
                        time.sleep(1.0)
                        print("Gripper closed")
                        print("\n*** OBJECT RECEIVED ***\n")
                        speak("Haha, gotcha! It's mine now!")
                        state = STATE_RETURN_HOME
                        hand_stable_start = None

                elif state == STATE_RETURN_HOME:
                    cv2.putText(display, "STATE: RETURNING TO HOME", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                    
                    speak("Taking this back to my spot.")
                    print("Returning to home position with object...")
                    go_to_custom_home(bot)
                    time.sleep(1.5)
                    print("At home position. Waiting for hand to enter frame...\n")
                    state = STATE_WAIT_HAND_ENTER

                elif state == STATE_WAIT_HAND_ENTER:
                    # Voice for entering this state
                    if spoken_state != STATE_WAIT_HAND_ENTER:
                        speak("Waiting for pickup...let me see your hand please")
                        spoken_state = STATE_WAIT_HAND_ENTER
                    
                    cv2.putText(display, "STATE: WAITING FOR HAND TO ENTER FRAME", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
                    cv2.putText(display, "Show your hand to receive the object", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    if hand_detected and hand_pos is not None:
                        print("Hand detected! Starting to track hand...")
                        speak("I see you, coming over.")
                        state = STATE_TRACK_HAND
                        hand_stable_start = None

                elif state == STATE_TRACK_HAND:
                    cv2.putText(display, "STATE: TRACKING HAND - Move hand to receive object", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
                    
                    if hand_detected and hand_pos is not None:
                        # Follow the hand
                        if is_reachable(hand_pos):
                            x_m = hand_pos[0] / 1000.0
                            y_m = hand_pos[1] / 1000.0
                            z_m = max(hand_pos[2] / 1000.0, 0.02)
                            bot.arm.set_ee_pose_components(x=x_m, y=y_m, z=z_m, moving_time=1.8)
                        
                        # Check hand stability
                        if hand_stable_start is None:
                            hand_stable_start = time.time()
                            print("Hand detected, checking stability...")
                        else:
                            stable_time = time.time() - hand_stable_start
                            cv2.putText(display, f"Hand stable: {stable_time:.1f}/{GIVE_HAND_STABLE_TIME:.1f}s", 
                                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            
                            if stable_time >= GIVE_HAND_STABLE_TIME:
                                # LOCK POSITION
                                locked_position = hand_pos.copy()
                                print("\n" + "="*50)
                                print("HAND STABLE - POSITION LOCKED!")
                                print("Waiting for human to pull object...")
                                print("="*50 + "\n")
                                speak("All yours. Give it a gentle tug.")
                                state = STATE_WAIT_FOR_PULL
                                hand_stable_start = None
                    else:
                        hand_stable_start = None

                elif state == STATE_WAIT_FOR_PULL:
                    cv2.putText(display, "STATE: POSITION LOCKED - Gripper holding, pull to release", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(display, "Pull the object gently...", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    # Get current effort reading
                    current_effort = get_gripper_effort(bot)
                    if current_effort is not None:
                        cv2.putText(display, f"Gripper effort: {current_effort:.2f}", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Set baseline on first entry to this state
                    if baseline_effort is None:
                        baseline_effort = current_effort
                        pull_start = None
                        if baseline_effort is not None:
                            print(f"  → Baseline effort set: {baseline_effort:.2f}")
                            print(f"  → Waiting for pull (threshold: Δ{EFFORT_THRESHOLD})...")
                    
                    # Check for pull - ONLY opens gripper when pull detected
                    if current_effort is not None and baseline_effort is not None:
                        effort_delta = abs(current_effort - baseline_effort)
                        
                        cv2.putText(display, f"Delta: {effort_delta:.2f} (threshold: {EFFORT_THRESHOLD})", 
                                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Print to terminal
                        print(f"  → Effort: {current_effort:.2f} (Δ{effort_delta:.2f})", end='\r')
                        
                        if effort_delta >= EFFORT_THRESHOLD:
                            if pull_start is None:
                                pull_start = time.time()
                                print(f"\n  → Pull detected! Δ{effort_delta:.2f}")
                            elif time.time() - pull_start >= EFFORT_STABLE_TIME:
                                print(f"\n  → Pull confirmed! Releasing now...")
                                speak("There you go!")
                                baseline_effort = None
                                pull_start = None
                                state = STATE_GIVE
                        else:
                            pull_start = None

                elif state == STATE_GIVE:
                    cv2.putText(display, "STATE: RELEASING OBJECT", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    
                    # Release object - only happens after pull detected
                    print("Opening gripper...")
                    bot.gripper.release(2.0)
                    time.sleep(1.0)
                    print("\n*** OBJECT GIVEN TO HUMAN ***\n")
                    
                    # Return to home
                    print("Returning to home position...")
                    time.sleep(0.5)
                    go_to_custom_home(bot)
                    time.sleep(1.0)
                    
                    # Close gripper and reset for next cycle
                    print("Closing gripper, ready for next handover...")
                    bot.gripper.grasp(2.0)
                    time.sleep(1.0)
                    
                    # Reset all state variables
                    state = STATE_FOLLOW
                    last_target = None
                    locked_position = None
                    baseline_effort = None
                    pull_start = None
                    hand_stable_start = None
                    spoken_state = None
                    spoken_hand_detected = False
                    
                    speak("That was fun! Let's go again.")
                    print("\n" + "="*50)
                    print("HANDOVER CYCLE COMPLETE!")
                    print("Ready for next handover cycle...")
                    print("="*50 + "\n")

                # Show mask view if enabled
                if show_mask:
                    mask = yellow_detector.get_mask(rgb_frame)
                    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    mask_colored[:,:,0] = 0  # Tint yellow
                    mask_colored[:,:,2] = 0
                    display = np.hstack((display, mask_colored))

                cv2.imshow("Proactive Handover", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('r'):
                    print("\n[RESET] Returning to home...")
                    go_to_custom_home(bot)
                    bot.gripper.grasp()
                    time.sleep(1.0)
                    state = STATE_FOLLOW
                    last_target = None
                    hand_stable_start = None
                    gripper_timer_start = None
                    locked_position = None
                    baseline_effort = None
                    pull_start = None
                    spoken_state = None
                    spoken_hand_detected = False
                    print("[RESET] Ready!\n")
                elif key == ord('o'):
                    print("\n[TEST] Opening gripper...")
                    bot.gripper.release()
                    time.sleep(1.0)
                    print("[TEST] Gripper opened!\n")
                elif key == ord('c'):
                    print("\n[TEST] Closing gripper...")
                    bot.gripper.set_pressure(1.0)
                    time.sleep(0.5)
                    bot.gripper.grasp()
                    time.sleep(1.0)
                    print("[TEST] Gripper closed!\n")
                elif key == ord('m'):
                    show_mask = not show_mask
                    print(f"[DEBUG] Mask view: {'ON' if show_mask else 'OFF'}")
                    if not show_mask:
                        cv2.destroyWindow("Proactive Handover")

        except KeyboardInterrupt:
            print("\n\nCtrl+C detected, shutting down...")

        print("Returning to sleep position...")
        bot.arm.go_to_sleep_pose()
        cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
