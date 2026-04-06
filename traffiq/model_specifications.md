# AI Model Specifications - Round 1

## Overview
Develop an AI model capable of visual processing and autonomous control for a robotic bot. 

### Core Requirements
The model must continuously process images to detect and interpret:
1. **Obstacles**: Placed on a black surface.
2. **Lane Tracking**: White lines along the center of a black track.
3. **Lighting**: Variations in lighting color.

## Model Input & Output
* **Input**: Image data from a Raspberry Pi Camera. The expected resolution is a minimum of `640 × 480 px` (final specification will be confirmed).
* **Output**: An array containing two continuous values representing the bot's desired movement commands: `[Speed, Direction]`. Both values must be scaled in the range `[-1, 1]`.

## Methodology & Architecture Considerations
* **Crucial Role of Object Detection**: Object detection is explicitly stated as being very important. The AI system must process the image and extract actionable information.
* **Flexibility in Approach**: The exact methodology/architecture you use to make these decisions from the imagery is entirely up to the team (e.g., Convolutional Neural Networks, YOLO, explicit image processing pipelines, etc.).

## Evaluation & Judging Criteria
For the first round, the organizers are focusing heavily on the development process and the team's comprehension of the system:
* **Understanding > Code Generation**: It is critical that the team builds the model and understands the code thoroughly. Judges will evaluate whether the team truly grasps the underlying implementation, rather than just copying AI-generated or pre-trained boilerplate code.
* **Process Matters**: The criteria care more about "how" the model was built than the specific output metrics on a given dataset.

## Hardware Support
* **Primary Sensor Data**: All critical data processed by this model will originate from a **Raspberry Pi Camera**.
