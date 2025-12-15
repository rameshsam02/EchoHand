# EchoHand: Proactive vs. Fixed-Position Robot Handover Systems

A research project investigating human-robot handover interactions by comparing proactive hand-tracking systems against traditional fixed-position approaches.

**CS 6983 – Human-Robot Interaction | Fall 2025**
**Northeastern University**

**Authors:** Samyukta Ramesh, Anusha Manohar, Caryn Dsouza

## Demo Video
[![EchoHand Demo](https://img.youtube.com/vi/HKD4DqPf8_M/0.jpg)](https://youtu.be/HKD4DqPf8_M)

---

## Overview

EchoHand explores whether a proactive handover system that dynamically tracks and responds to human hand position can create more natural exchanges compared to traditional fixed-position handovers. The project implements two complete bidirectional handover systems using an RX200 robotic arm with OAK-D Lite depth camera integration.

## Abstract

Human-robot handover is a fundamental interaction in collaborative robotics, yet most robotic systems rely on reactive, pre-programmed behaviors that create awkward pauses and reduce user trust. This project investigates proactive handover systems through rigorous coordinate transformation mathematics to enable accurate camera-to-robot spatial reasoning.

### Key Findings

A within-subjects user study with five participants revealed:

- **Fixed-position system**: 93.3% success rate for both give and receive tasks
- **Proactive system**: Comparable subjective ratings despite lower receive success rates (53.3%)
- Both systems rated favorably for safety, naturalness, and efficiency
- Fixed-position system showed advantages in predictability and timing

## System Design

### Hardware Platform

- **Robot**: Interbotix RX200 robotic arm (5-DOF manipulator)
- **Vision**: OAK-D Lite depth camera
  - Positioned at 580mm height
  - 50-degree downward tilt
  - Approximately 700mm from robot base
- **Test Object**: Yellow cuboid (selected for reliable graspability and visual detection)

### Software Architecture

- **Framework**: Python with ROS2 Galactic
- **Hand Tracking**: Google's MediaPipe Hands + OAK-D Lite depth data for 3D positioning
- **Coordinate Transform**: Custom matrix for camera-to-robot frame conversion
- **Gripper Control**: DYNAMIXEL servo effort feedback (threshold ~10.0 for grasp detection)
- **User Feedback**: Text-to-speech using Google's gTTS library

### System Variants

1. **Fixed-Position System**

   - Predefined handover locations
   - Higher reliability (93.3% success rate)
   - Better predictability and timing

2. **Proactive System**
   - Dynamic hand tracking
   - Real-time position adaptation
   - More natural interaction qualities

## Research Implications

The findings suggest that proactive handover systems offer promising interaction qualities but require further refinement in:

- Stability detection algorithms
- Coordinate transform accuracy
- Real-time response consistency

## Applications

- **Assistive Robotics**: Smooth handovers critical for user acceptance
- **Manufacturing**: Efficient handovers impact productivity
- **Service Robotics**: Improved worker satisfaction through natural interactions

## Challenges

The proactive system demonstrated excellent performance when successfully reaching the human's hand, with smooth and natural handover execution. Key challenges included:

- **Coordinate Transform Accuracy**: Initial positioning errors of 10-15cm (with Z-errors up to 60cm in early iterations) were reduced to 3-5cm through systematic calibration. Residual errors impacted proactive receive success rates.

- **Stability Detection**: Balancing sensitivity against false triggers in the 5cm stability zone. Occasional premature locking or missed genuine stability signals caused mistimed gripper actions.

- **Current Sensing Limitations**: Effort sensing worked well for detecting pulls on held objects but was unreliable for detecting lightweight objects placed in open grippers, necessitating different strategies for give (current-based) versus receive (distance-based hand retreat detection).

- **Computational Constraints**: YOLO caused substantial lag; HSV thresholding resolved this but limited detection to specifically colored objects under controlled lighting.

Despite these challenges, the 93.3% give success rate demonstrates core mechanisms function effectively. The 53.3% receive rate reflects positioning and stability detection challenges rather than fundamental paradigm flaws.

## Future Work

Several directions for future development:

- **Automated Calibration**: Employ fiducial markers (ArUco tags) or implement online refinement that continuously adjusts parameters based on observed errors
- **Enhanced Stability Detection**: Predictive filtering techniques (Kalman filtering) for more robust hand position and velocity estimation
- **Safety Measures**: Force limiting, collision detection, and accessible emergency stops for real-world deployment
- **Improved Object Detection**: Expand beyond HSV thresholding to lightweight neural networks for better generalizability
- **Larger-Scale Studies**: Validate trends with diverse populations, including elderly users and individuals with motor impairments

## Installation & Setup

### Prerequisites

- ROS2 Galactic
- Interbotix RX200 robot workspace
- Python 3 with required dependencies:
  - MediaPipe Hands
  - Google gTTS
  - OAK-D Lite camera drivers

### Hardware Setup

**Terminal 1: Robot Control**

```bash
# Check and reset port (run every time)
ls /dev | grep ttyDXL

# Launch robot control
ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=rx200
```

**Terminal 2: Run System**

```bash
# Navigate to demos directory
cd ~/interbotix_ws/src/interbotix_ros_manipulators/interbotix_ros_xsarms/interbotix_xsarm_control/demos/python_ros2_api

# Run fixed-position system
python3 phase1.py

# OR run proactive system
python3 phase2.py
```

### Simulation Setup

**Terminal 1: Launch Simulator**

```bash
source /opt/ros/galactic/setup.bash
source ~/interbotix_ws/install/setup.bash

ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=rx200 use_sim:=true
```

**Terminal 2: Run System**

```bash
# Run fixed-position system
python3 phase1.py

# OR run proactive system
python3 phase2.py
```

## Usage

The system supports two modes of operation:

### Fixed-Position System ([phase1.py](phase1.py))

- Predefined handover locations
- Higher reliability for consistent environments
- Recommended for initial testing and stable applications

### Proactive System ([phase2.py](phase2.py))

- Dynamic hand tracking with real-time adaptation
- More natural interaction experience
- Requires proper calibration for optimal performance

Both systems provide:

- **Bidirectional handovers**: Robot can both give and receive objects
- **Audio feedback**: Text-to-speech status updates
- **Visual tracking**: Real-time hand position monitoring
- **Gripper control**: Automatic grasp detection and release

## Acknowledgments

This project was completed as part of CS 6983 – Human-Robot Interaction at Northeastern University, Fall 2025.

---

**Project Type:** Academic Research

**Status:** Completed

**Course:** CS 6983 – Human-Robot Interaction
