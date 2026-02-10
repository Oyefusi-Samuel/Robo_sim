# HRI Precise Cube Sorter: Supervisory Control Simulation

This project, developed for **RBE 526**, demonstrates a sophisticated **Human-Robot Interaction (HRI)** system. It features a Franka Emika Panda robot performing a semi-autonomous sorting task where a human supervisor provides real-time authorization via computer vision gestures.

---

## Project Overview
The system bridges the gap between full autonomy and manual teleoperation. The robot identifies and plans paths to 8 mixed-color cubes, but it is programmed with a "safety interlock": it will not initiate a pick-and-place sequence without a **Thumbs Up** confirmation from the human supervisor.

---

## 🛠 Key Features
* **Supervisory Control Logic:** The robot manages complex Inverse Kinematics (IK) and multi-stage path planning while remaining under human oversight.
* **Gesture Recognition:** Integrated **MediaPipe** pipeline to detect hand landmarks and interpret a "Thumbs Up" as a binary approval signal.
* **Precise State Machine:** A robust 6-stage logic flow:
    1. **APPROACHING**: High-level horizontal travel at a `SAFE_HEIGHT` to prevent collisions.
    2. **DESCENDING**: Vertical approach to the target object upon human approval.
    3. **PICKING**: Fixed-constraint grasping for stable object manipulation.
    4. **LIFTING**: Vertical extraction to clear the workspace.
    5. **MOVING**: Lateral transport to the designated tray.
    6. **DROPPING**: Low-velocity "soft landing" release.
* **Physics-Engine Integration:** Built on **PyBullet** with realistic gravity, mass dynamics, and collision meshes.

---

## 📐 System Architecture

### 1. The HRI Interface
The system uses the webcam to create a feedback loop.
* **Approval Gesture:** The `is_thumbs_up()` function monitors the $y$-coordinates of hand landmarks. Approval is triggered when the thumb tip is significantly higher than all other fingertips.
* **Visual HUD:** A real-time Head-Up Display (HUD) is overlaid on the video feed, showing the robot's current state and required supervisor actions.

### 2. Workspace Calibration
* **Robot Base:** Mounted at `[0, 0, 0.62]` to sit flush on the table surface.
* **Trays:** Blue (Left) and Red (Right) trays are modeled as 3D containers with specific coordinate bounds for scoring.
* **Cubes:** 8 cubes are spawned with randomized positions and colors to test the robot's adaptability.

---

## 💻 Installation & Usage

### Prerequisites
Ensure you have Python installed with the following dependencies:
```bash
pip install pybullet opencv-python mediapipe numpy


