# Lab 3: GenAI for HRI

## 1. Introduction

### 1.1 Problem Statement and Scope
#### HRI Problem, Application, and LLM/VLM/VLA Scope

Hands-on building activities, such as assembling LEGO models, are a core part of middle school STEM education. They help students develop spatial reasoning, follow multi-step instructions, and learn engineering concepts through direct experience. These activities depend on timely feedback, though: a mistake in an early step, like a misplaced brick, often goes unnoticed until several steps later, when it is harder to fix. In a typical classroom, a single teacher supervises many groups at once and cannot check each student’s progress at every step. We will address this problem by developing a desktop robot coach that watches a student build and checks each step as it is completed. Our system uses a Reachy Mini robot with a camera and a Vision-Language Model (VLM). After the student finishes a step, the VLM compares an image of the current build against the reference instructions for that step. The robot then responds with speech and expressive head and antenna gestures, either confirming the step or pointing out the mistake. Our scope focuses on the end-to-end behavior of the robot, rather than on the VLM in isolation: whether it correctly verifies each step, whether it catches mistakes before the student moves on, and how quickly it responds.

#### Motivation and Importance to the HRI Community

Addressing this problem matters for both students and the HRI community. Middle school students are often reluctant to ask for help in front of their peers, so errors that a teacher does not notice may simply persist. A robot that monitors progress and offers feedback without being asked could reduce frustration and keep students engaged. For the HRI community, this application raises open questions about grounding visual perception in physical, real-world tasks. Prior work on robots in education has largely focused on conversational tutoring, while VLMs have mostly been evaluated on static image benchmarks rather than in real-time interactions with physical objects [CITE]. A robot coach for building tasks must judge small spatial details, like a brick’s color, position, and orientation, under real classroom conditions with changing lighting, camera angles, and partially hidden pieces. It must also communicate its judgments in a way a student will trust and act on, since false alarms and missed mistakes can both undermine the student’s confidence in the robot. Studying how VLM perception performs as part of an embodied, end-to-end robot system helps show whether today’s VLMs are ready to support hands-on learning, and where they still fall short.

#### Video Demo

### 1.2 Research Questions / Hypotheses
#### RQ1 / H1

RQ1: How accurately does the robot coach verify each building step, and which kinds of student mistakes does it miss most often?

#### RQ2 / H2

H2: The robot will verify steps more accurately when the VLM gets a reference image of the correct step than when it gets only a text description of that step

## 2. Background

### 2.1 Theme 1
### 2.2 Theme 2
### 2.3 Research Gaps and Novelty

## 3. Robot Control Architecture

### 3.1 Robot Goals
#### Robot Goals and System-Focused Problem
#### Interaction Design Motivation
#### Assessment Through Objective Measures

### 3.2 Architecture Design
#### Inputs
#### Outputs
#### Modules
#### Architecture Type and Justification
#### Architecture Figure

## 4. Evaluation

### 4.1 Study Participants
#### Number of Participants
#### Sampling Method

### 4.2 Experimental Design
#### Independent Variables
#### Dependent Variables
#### Study Design
#### Number of Trials
#### How the Comparison Addresses the RQs / Hypotheses
#### Evaluation Workflow Figure

### 4.3 Study Task
#### End-to-End HRI Task
#### Level of Autonomy and Human Intervention

### 4.4 Data Collection and Analysis
#### Data Sources and Types
#### Number of Observations
#### Analysis Methods

### 4.5 Metrics
#### Objective Measures

## 5. Results

### 5.1 Outcomes by Dependent Variable
### 5.2 Best and Worst Conditions
### 5.3 Patterns Across Conditions

## 6. Discussion

### 6.1 Theme 1
### 6.2 Theme 2
### 6.3 Implications for RQs / Hypotheses
### 6.4 Comparison with Prior Work
### 6.5 Contributions and Implications

## 7. Limitations and Future Work

### 7.1 Limitation 1
### 7.2 Limitation 2
### 7.3 Future Work

## 8. Code

### Setup
### Running the Demos

## References
