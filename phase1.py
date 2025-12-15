
#!/usr/bin/env python3
# RX200 Current-Based Handover with TTS Responses
# Uses gripper EFFORT sensing to detect user grabbing

from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import numpy as np
import sys
import tty
import termios
import time
import cv2
import mediapipe as mp
import threading
import depthai as dai
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
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                    temp_path = fp.name
                
                tts = gTTS(text=text, lang='en')
                tts.save(temp_path)
                
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                os.unlink(temp_path)
            except Exception as e:
                print(f"[TTS Error] {e}")
    
    thread = threading.Thread(target=_speak, daemon=True)
    thread.start()
# =========================================================

"""
Improved Handover Sequence:
  1) Custom Home (compact position, open gripper)
  2) Rotate waist ±45°
  3) WAIT for hand near gripper → OPEN → hand delivers → GRIP IMMEDIATELY
  4) Rotate waist back to 0°
  5) WAIT for hand near gripper → SENSE USER GRAB (effort) → OPEN automatically
  6) Maintain Custom Home

Controls:
  SPACE : run sequence
  L     : set direction LEFT  (-45°)
  R     : set direction RIGHT (+45°)
  V     : toggle vision debug window
  T     : test gripper sensing
  Q     : quit
"""

# ---- Tunables --------------------------------------------------------------
TURN_DEG = 45.0            # waist rotation magnitude
SETTLE = 0.8               # seconds after each move

# Custom Home Position
CUSTOM_HOME_POSITION = {
    'waist': 0.000,
    'shoulder': -0.800,
    'elbow': 1.300,
    'wrist_angle': 0.500,
    'wrist_rotate': 0.000
}

# Hand-to-Gripper proximity thresholds (3D Euclidean distance)
PROXIMITY_THRESHOLD = 0.30  # meters (30cm) - hand must be this close to gripper
STABLE_TIME = 0.5          # seconds hand must be stable
TIMEOUT_SEC = 20.0         # max wait time

# Gripper EFFORT sensing (THIS IS WHAT WORKS!)
GRAB_EFFORT_THRESHOLD = 10.0  # Effort change when user grabs/pulls
GRAB_STABLE_TIME = 0.3        # Seconds grip must be stable

# Gripper detection (color-based - adjust for your gripper color)
GRIPPER_COLOR_LOWER = np.array([0, 0, 0])      # HSV lower bound
GRIPPER_COLOR_UPPER = np.array([180, 255, 80])  # HSV upper bound
GRIPPER_MIN_AREA = 1000     # minimum contour area for gripper

# WINDOW SETTINGS
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720
# ----------------------------------------------------------------------------

class OAKDHandTracker:
    """Real-time hand + gripper tracking with OAK-D Lite"""
    
    def __init__(self):
        # Initialize OAK-D pipeline
        self.pipeline = dai.Pipeline()
        
        # RGB camera
        cam_rgb = self.pipeline.create(dai.node.ColorCamera)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        cam_rgb.setFps(30)
        
        # Depth camera
        mono_left = self.pipeline.create(dai.node.MonoCamera)
        mono_right = self.pipeline.create(dai.node.MonoCamera)
        stereo = self.pipeline.create(dai.node.StereoDepth)
        
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        
        # Output streams
        xout_rgb = self.pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.video.link(xout_rgb.input)
        
        xout_depth = self.pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)
        
        # MediaPipe hands - IMPROVED SETTINGS
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            model_complexity=1
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Gripper tracking
        self.gripper_detected = False
        self.gripper_pos_2d = None
        self.gripper_pos_3d = None
        
        # Hand tracking
        self.hand_detected = False
        self.hand_near_gripper = False
        self.hand_center_2d = None
        self.hand_center_3d = None
        self.distance_to_gripper = float('inf')
        
        self.frame = None
        self.depth_frame = None
        self.running = False
        self.show_debug = False
        self.device = None
        self.window_initialized = False
        
    def start(self):
        """Start tracking thread"""
        self.device = dai.Device(self.pipeline)
        self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        self.q_depth = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        
        self.running = True
        self.thread = threading.Thread(target=self._track_loop, daemon=True)
        self.thread.start()
        time.sleep(2.0)
        print("[INFO] OAK-D Lite camera initialized!")
        print("[INFO] Automatic gripper detection enabled")
        
    def stop(self):
        """Stop tracking thread"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2.0)
        if self.device:
            self.device.close()
        cv2.destroyAllWindows()
        
    def detect_gripper(self, frame, depth_frame):
        """Detect gripper using color-based detection"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for gripper color
        mask = cv2.inRange(hsv, GRIPPER_COLOR_LOWER, GRIPPER_COLOR_UPPER)
        
        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (assumed to be gripper)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > GRIPPER_MIN_AREA:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Gripper center in pixels
                center_x = x + w // 2
                center_y = y + h // 2
                
                self.gripper_pos_2d = (center_x, center_y)
                
                # Get 3D position
                self.gripper_pos_3d = self.get_3d_position(center_x, center_y, depth_frame)
                
                self.gripper_detected = True
                return (x, y, w, h), mask
        
        self.gripper_detected = False
        return None, mask
        
    def get_3d_position(self, x_pixel, y_pixel, depth_frame):
        """Convert 2D pixel + depth to 3D position"""
        if depth_frame is None or x_pixel < 0 or y_pixel < 0:
            return None
            
        h, w = depth_frame.shape
        if y_pixel >= h or x_pixel >= w:
            return None
            
        # Get depth at point (with region averaging for stability)
        region_size = 5
        y_min = max(0, int(y_pixel) - region_size)
        y_max = min(h, int(y_pixel) + region_size)
        x_min = max(0, int(x_pixel) - region_size)
        x_max = min(w, int(x_pixel) + region_size)
        
        depth_region = depth_frame[y_min:y_max, x_min:x_max]
        valid_depths = depth_region[depth_region > 0]
        
        if len(valid_depths) == 0:
            return None
            
        z_mm = np.median(valid_depths)
        z_m = z_mm / 1000.0
        
        # Convert to 3D (approximate using pinhole model)
        x_normalized = (x_pixel - w/2) / w
        y_normalized = (y_pixel - h/2) / h
        
        x_m = x_normalized * z_m
        y_m = y_normalized * z_m
        
        return (x_m, y_m, z_m)
        
    def _track_loop(self):
        """Main tracking loop"""
        while self.running:
            # Get RGB frame
            in_rgb = self.q_rgb.get()
            frame = in_rgb.getCvFrame()
            
            # Get depth frame
            in_depth = self.q_depth.get()
            depth_frame = in_depth.getFrame()
            
            self.depth_frame = depth_frame
            h, w, _ = frame.shape
            
            # 1. DETECT GRIPPER automatically
            gripper_bbox, gripper_mask = self.detect_gripper(frame, depth_frame)
            
            # Draw gripper if detected
            if gripper_bbox and self.gripper_pos_2d:
                x, y, gw, gh = gripper_bbox
                cv2.rectangle(frame, (x, y), (x + gw, y + gh), (255, 0, 0), 2)
                cv2.circle(frame, self.gripper_pos_2d, 8, (255, 0, 0), -1)
                cv2.putText(frame, "GRIPPER", (x, y - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # 2. DETECT HAND using MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            # Reset hand detection
            self.hand_detected = False
            self.hand_near_gripper = False
            self.hand_center_2d = None
            self.hand_center_3d = None
            self.distance_to_gripper = float('inf')
            
            if results.multi_hand_landmarks:
                # Use the first detected hand
                hand_lm = results.multi_hand_landmarks[0]
                
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    frame, hand_lm, self.mp_hands.HAND_CONNECTIONS
                )
                
                # Get hand center (palm base)
                palm_landmarks = [hand_lm.landmark[i] for i in [0, 5, 17]]
                palm_x = sum([lm.x for lm in palm_landmarks]) / 3
                palm_y = sum([lm.y for lm in palm_landmarks]) / 3
                
                hand_center_px = (int(palm_x * w), int(palm_y * h))
                self.hand_center_2d = hand_center_px
                
                # Get 3D position of hand
                self.hand_center_3d = self.get_3d_position(
                    hand_center_px[0], hand_center_px[1], depth_frame
                )
                
                if self.hand_center_3d:
                    self.hand_detected = True
                    
                    # Draw hand center
                    cv2.circle(frame, hand_center_px, 8, (0, 255, 255), -1)
                    
                    # 3. CALCULATE HAND-TO-GRIPPER DISTANCE
                    if self.gripper_detected and self.gripper_pos_3d:
                        dx = self.hand_center_3d[0] - self.gripper_pos_3d[0]
                        dy = self.hand_center_3d[1] - self.gripper_pos_3d[1]
                        dz = self.hand_center_3d[2] - self.gripper_pos_3d[2]
                        self.distance_to_gripper = np.sqrt(dx**2 + dy**2 + dz**2)
                        
                        # Check proximity
                        if self.distance_to_gripper < PROXIMITY_THRESHOLD:
                            self.hand_near_gripper = True
                            color = (0, 255, 0)  # Green
                            status = "CLOSE TO GRIPPER!"
                        else:
                            color = (0, 165, 255)  # Orange
                            status = "Hand detected"
                        
                        # Draw connection line
                        cv2.line(frame, hand_center_px, self.gripper_pos_2d, color, 2)
                        
                        # Display distance
                        mid_x = (hand_center_px[0] + self.gripper_pos_2d[0]) // 2
                        mid_y = (hand_center_px[1] + self.gripper_pos_2d[1]) // 2
                        
                        distance_text = f"{self.distance_to_gripper * 100:.1f} cm"
                        cv2.putText(frame, distance_text, (mid_x - 50, mid_y - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                        
                        # Status text
                        cv2.putText(frame, status, (10, 40),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                    else:
                        cv2.putText(frame, "Hand detected (no gripper)", (10, 40),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Display detection status
            if not self.gripper_detected:
                cv2.putText(frame, "Searching for gripper...", (10, h - 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if not self.hand_detected:
                cv2.putText(frame, "No hand detected", (10, 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Display threshold
            cv2.putText(frame, f"Proximity threshold: {PROXIMITY_THRESHOLD*100:.0f}cm",
                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            self.frame = frame
            
            # Show debug window if enabled
            if self.show_debug:
                if not self.window_initialized:
                    cv2.namedWindow('Tracking (RGB + Gripper Mask)', cv2.WINDOW_NORMAL)
                    combined_width = WINDOW_WIDTH * 2
                    cv2.resizeWindow('Tracking (RGB + Gripper Mask)', 
                                   combined_width, WINDOW_HEIGHT)
                    self.window_initialized = True
                
                mask_3ch = cv2.cvtColor(gripper_mask, cv2.COLOR_GRAY2BGR)
                combined = np.hstack((frame, mask_3ch))
                cv2.imshow('Tracking (RGB + Gripper Mask)', combined)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.show_debug = False
                    self.window_initialized = False
                    cv2.destroyAllWindows()
            else:
                if self.window_initialized:
                    self.window_initialized = False
                    cv2.destroyAllWindows()
    
    def wait_for_hand_near_gripper(self, timeout=TIMEOUT_SEC, speak_waiting=True):
        """Wait for hand to approach gripper"""
        print(f"  → Waiting for hand near gripper (< {PROXIMITY_THRESHOLD*100:.0f}cm)...")
        if speak_waiting:
            speak("Waiting for your hand.")
        
        start_time = time.time()
        stable_start = None
        hand_announced = False
        
        while time.time() - start_time < timeout:
            if not self.gripper_detected:
                print("  → Warning: Gripper not detected!")
                time.sleep(0.5)
                continue
                
            if self.hand_near_gripper:
                if stable_start is None:
                    stable_start = time.time()
                    print(f"  → Hand approaching! Distance: {self.distance_to_gripper*100:.1f}cm")
                    if not hand_announced:
                        speak("I see your hand.")
                        hand_announced = True
                elif time.time() - stable_start >= STABLE_TIME:
                    print(f"  → Hand stable near gripper! ✓ ({self.distance_to_gripper*100:.1f}cm)")
                    return True
            else:
                if self.hand_detected:
                    print(f"  → Hand detected: {self.distance_to_gripper*100:.1f}cm away...")
                stable_start = None
            time.sleep(0.1)
        
        print(f"  → Timeout!")
        speak("No hand detected, skipping.")
        return False


def test_gripper_sensing(bot):
    """Quick test of gripper effort sensing"""
    print("\n" + "="*60)
    print("GRIPPER EFFORT TEST")
    print("="*60)
    
    speak("Starting gripper test.")
    
    print("\n[Test] Closing with object - Place object NOW!")
    bot.gripper.release()
    time.sleep(2.5)
    bot.gripper.grasp()
    time.sleep(1.0)
    
    print("  Baseline (holding object):")
    try:
        js = bot.gripper.core.joint_states
        if js and hasattr(js, 'effort') and len(js.effort) > 0:
            baseline_effort = js.effort[0] if hasattr(js.effort, '__getitem__') else js.effort
            print(f"  Effort: {baseline_effort:.2f}")
        else:
            baseline_effort = 0
    except:
        baseline_effort = 0
        print("  Effort: N/A")
    
    print("\n[PULL the object NOW for 10 seconds!]")
    speak("Pull the object now.")
    for i in range(20):
        try:
            js = bot.gripper.core.joint_states
            if js and hasattr(js, 'effort') and len(js.effort) > 0:
                effort = js.effort[0] if hasattr(js.effort, '__getitem__') else js.effort
                effort_change = abs(effort - baseline_effort)
                
                if effort_change >= GRAB_EFFORT_THRESHOLD:
                    print(f"  ✓ GRIP DETECTED! Effort: {effort:.2f} (Δ{effort_change:.2f})")
                else:
                    print(f"  Effort: {effort:.2f} (Δ{effort_change:.2f})")
        except Exception as e:
            print(f"  Error: {e}")
        
        time.sleep(0.5)
    
    speak("Gripper test complete.")
    print("\n" + "="*60)


def wait_for_user_grab(bot, timeout=TIMEOUT_SEC):
    """Detect grab by effort increase - THE WORKING METHOD!"""
    print(f"  → Using EFFORT-based detection (threshold: {GRAB_EFFORT_THRESHOLD})...")
    
    try:
        js = bot.gripper.core.joint_states
        if js and hasattr(js, 'effort') and len(js.effort) > 0:
            baseline_effort = js.effort[0] if hasattr(js.effort, '__getitem__') else js.effort
            print(f"  → Baseline effort: {baseline_effort:.2f}")
        else:
            print("[ERROR] Cannot read gripper effort")
            return False
    except Exception as e:
        print(f"[ERROR] Cannot read gripper effort: {e}")
        return False
    
    start_time = time.time()
    stable_start = None
    
    while time.time() - start_time < timeout:
        try:
            js = bot.gripper.core.joint_states
            if js and hasattr(js, 'effort') and len(js.effort) > 0:
                current_effort = js.effort[0] if hasattr(js.effort, '__getitem__') else js.effort
                effort_change = abs(current_effort - baseline_effort)
                
                print(f"  → Effort: {current_effort:.2f} (Δ{effort_change:.2f})", end='\r')
                
                if effort_change >= GRAB_EFFORT_THRESHOLD:
                    if stable_start is None:
                        stable_start = time.time()
                        print(f"\n  → Grip detected! Effort: {current_effort:.2f}")
                    elif time.time() - stable_start >= GRAB_STABLE_TIME:
                        print(f"\n  → Grip confirmed! ✓")
                        speak("Releasing now, there you go!")
                        return True
                else:
                    stable_start = None
        except:
            pass
        
        time.sleep(0.1)
    
    print(f"\n  → Timeout")
    speak("Didn't feel a pull, opening anyway.")
    return False


def get_key():
    """Get a single keypress without Enter."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def radians(deg): 
    return deg * np.pi / 180.0


def go_to_custom_home(bot):
    """Move to custom compact home position"""
    bot.arm.set_joint_positions(
        [CUSTOM_HOME_POSITION['waist'],
         CUSTOM_HOME_POSITION['shoulder'],
         CUSTOM_HOME_POSITION['elbow'],
         CUSTOM_HOME_POSITION['wrist_angle'],
         CUSTOM_HOME_POSITION['wrist_rotate']]
    )


def main():
    # Initialize robot
    bot = InterbotixManipulatorXS(
        robot_model='rx200',
        group_name='arm',
        gripper_name='gripper'
    )
    
    # Initialize OAK-D tracker
    print("[INFO] Initializing OAK-D Lite camera...")
    tracker = OAKDHandTracker()
    tracker.start()
    
    print("\n" + "="*60)
    print(" RX200 — Smart Grip Detection Handover with TTS")
    print("="*60)
    print("Controls:")
    print("  L     : set turn LEFT  (-45°)")
    print("  R     : set turn RIGHT (+45°)")
    print("  V     : toggle vision debug window")
    print("  T     : test gripper effort sensing")
    print("  SPACE : run handover sequence")
    print("  Q     : quit\n")
    print("[INFO] Detection: EFFORT-based (WORKING!)")
    print(f"[INFO] Proximity threshold: {PROXIMITY_THRESHOLD*100:.0f}cm")
    print(f"[INFO] Effort threshold: {GRAB_EFFORT_THRESHOLD}\n")
    
    # Default direction: RIGHT
    dir_sign = +1
    print("[INFO] Default turn direction: RIGHT (+45°).\n")
    
    # Move to home
    print("[INFO] Moving to compact home position...")
    go_to_custom_home(bot)
    bot.gripper.release()
    time.sleep(SETTLE)
    
    # Startup greeting
    speak("Beep boop... just kidding. I'm ready for handovers!")
    
    try:
        while True:
            print("\nWaiting for key... (L/R/V/T/SPACE/Q)")
            key = get_key()
            
            if key in ('q', 'Q', '\x03'):
                print("\n[INFO] Quit requested.")
                speak("Goodbye!")
                break
            
            if key in ('l', 'L'):
                dir_sign = -1
                print("[INFO] Turn direction: LEFT (-45°)")
                speak("Turning left next time.")
                continue
            
            if key in ('r', 'R'):
                dir_sign = +1
                print("[INFO] Turn direction: RIGHT (+45°)")
                speak("Turning right next time.")
                continue
            
            if key in ('v', 'V'):
                tracker.show_debug = not tracker.show_debug
                if tracker.show_debug:
                    print("[INFO] Debug window enabled")
                else:
                    print("[INFO] Debug window disabled")
                continue
            
            if key in ('t', 'T'):
                test_gripper_sensing(bot)
                continue
            
            if key != ' ':
                continue
            
            # ============== RUN HANDOVER SEQUENCE ==================
            try:
                print("\n" + "="*60)
                print("STARTING HANDOVER SEQUENCE")
                print("="*60)
                speak("Starting handover sequence.")
                
                if not tracker.gripper_detected:
                    print("[WARNING] Gripper not detected! Continuing anyway...")
                
                waist_start = float(bot.arm.get_single_joint_command('waist'))
                
                # 1) Home
                print("\n[Step 1/6] At home position...")
                go_to_custom_home(bot)
                bot.gripper.release()
                time.sleep(SETTLE)
                
                # 2) Rotate waist
                turn_rad = dir_sign * radians(TURN_DEG)
                direction = 'RIGHT' if dir_sign > 0 else 'LEFT'
                print(f"\n[Step 2/6] Rotating waist {direction}...")
                speak("Rotating to receive position.")
                bot.arm.set_single_joint_position('waist', waist_start + turn_rad)
                time.sleep(SETTLE)
                
                # 3) RECEIVE object
                print("\n[Step 3/6] HANDOVER RECEIVE")
                print("=" * 60)
                
                if tracker.wait_for_hand_near_gripper(timeout=TIMEOUT_SEC):
                    print("  → Hand detected near gripper!")
                    print("  → Opening gripper...")
                    speak("Opening gripper.")
                    bot.gripper.release()
                    time.sleep(0.5)
                    
                    print("  → Place object in gripper...")
                    speak("Place the object now.")
                    time.sleep(0.8)
                    
                    print("  → Closing gripper now!")
                    bot.gripper.grasp()
                    time.sleep(SETTLE)
                    print("  → Object grasped! ✓")
                    speak("Got it!")
                else:
                    print("[WARNING] No hand detected. Skipping receive.")
                
                print("=" * 60)
                
                # 4) Return waist
                print("\n[Step 4/6] Returning waist to center...")
                speak("Returning to center.")
                bot.arm.set_single_joint_position('waist', waist_start)
                time.sleep(SETTLE)
                
                # 5) GIVE object - EFFORT SENSING!
                print("\n[Step 5/6] HANDOVER GIVE")
                print("=" * 60)
                speak("Ready to hand over, show me your hand.")
                
                if tracker.wait_for_hand_near_gripper(timeout=TIMEOUT_SEC, speak_waiting=False):
                    print("  → Hand detected near gripper!")
                    speak("Hand detected, grab and pull when ready.")
                    print("  → Please GRAB and PULL the object...")
                    
                    if wait_for_user_grab(bot, timeout=TIMEOUT_SEC):
                        print("  → Opening gripper NOW!")
                        bot.gripper.release()
                        time.sleep(0.5)
                        print("  → Object released! ✓")
                    else:
                        print("[WARNING] No grip detected. Opening anyway...")
                        bot.gripper.release()
                        time.sleep(SETTLE)
                else:
                    print("[WARNING] No hand detected. Opening gripper anyway...")
                    bot.gripper.release()
                    time.sleep(SETTLE)
                
                print("=" * 60)
                
                # 6) Done
                print("\n[Step 6/6] Complete!")
                time.sleep(SETTLE)
                
                print("\n" + "="*60)
                print("SEQUENCE COMPLETE!")
                print("="*60)
                speak("Handover complete! Ready for another round.")
                
            except Exception as e:
                print(f"\n[ERROR] {type(e).__name__}: {e}")
                print("Returning home...")
                try:
                    go_to_custom_home(bot)
                except:
                    pass
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    
    finally:
        print("\n[INFO] Shutting down...")
        tracker.stop()
        try:
            bot.arm.go_to_sleep_pose()
        except:
            pass
        bot.shutdown()
        print("[INFO] Goodbye!")


if __name__ == "__main__":
    main()
