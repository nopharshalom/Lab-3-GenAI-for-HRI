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

### 1.3 HRI Literature Review and Research Gaps

#### Social Robots and Generative AI for Children’s Learning

Social robots have been widely studied as tutors and learning companions for children, and recent work has focused on how a robot’s behavior can support learning tasks. Ligthart et al. [1] designed a social robot math tutor and tested it in a three-session study with 130 children aged 8–11 (HRI 2023). Children improved at math over time when the robot scaffolded its instructions, and they saw the robot more as a friend when it personalized the conversation. More recently, researchers have begun adding LLMs to child-robot interaction. Rudenko et al. [2] studied the linguistic and interactive quality of language produced by an LLM-powered social robot for preschoolers and emphasized the need to validate LLM outputs before using them with children. Fernandes et al. [3] ran a co-design study with tutors and program coordinators for a classroom robot, noting that literacy programs are often limited by small staff, mixed-proficiency groups, and short contact time. This mirrors the supervision problem in hands-on STEM classrooms. Across this work, the robot mostly supports learning through conversation; the child’s learning activity happens in speech or on a screen rather than with physical objects.

#### Vision-Language Models for Task Verification and Error Detection

A second line of work uses foundation models to check whether a physical task was done correctly. REFLECT [4] (CoRL 2023) queries an LLM to reason about failures using a hierarchical summary of the robot’s past experiences built from multisensory observations, and the resulting explanation guides a planner to correct the failure. AHA [5] treats failure detection as a free-form reasoning task, so a VLM both identifies failures and explains them in natural language across different robots and tasks, and its feedback improved sub-task verification and raised task success by an average of 21.4%. Guardian [6] notes that VLM accuracy and generalization for this problem are limited by the scarcity of failure data. Closest to our task, Kottom et al. [7] built an AR headset system for LEGO assembly in which each step is validated by a VLM so that user mistakes are detected and the user gets feedback on how to return to a valid state. Together, these papers show that VLMs can verify individual steps of a physical task, though their reliability on fine-grained spatial details remains an open problem.

#### Research Gaps

Our work addresses three gaps across these themes. First, most VLM-based failure detection, including REFLECT, AHA, and Guardian, checks the robot’s own manipulation errors. Our robot instead would check the human’s work, which introduces new challenges like hands blocking the view and builds that vary from student to student. Second, child-robot interaction research has largely focused on conversational tutoring [1, 2], and has rarely studied robots that give feedback on a child’s hands-on physical activity. Third, the closest system for step-by-step LEGO verification [7] delivers feedback through an AR headset rather than through an embodied social robot. Headsets are expensive and impractical for middle school classrooms, and they lose the social cues, like gaze and gestures, that make a robot feel like a learning partner. Our project combines these threads by evaluating the end-to-end performance of a low-cost desktop social robot that uses a VLM to verify each step of a student’s build and responds with speech and expressive gestures.

#### References
[1] M. Ligthart, S. de Droog, M. Bossema, L. Elloumi, K. Hoogland, M. Smakman, K. Hindriks, S. Ben Allouch. “Design specifications for a social robot math tutor.” HRI ‘23, pp. 321–330, 2023. doi:10.1145/3568162.3576957
[2] Rudenko et al. LLM-powered social robot for preschoolers. HRI Companion ‘26, 2026. (Look up the full title and author list on Google Scholar.)
[3] N. Fernandes et al. “Co-designing a Social Robot for Newcomer Children’s Cultural and Language Learning.” HRI 2026.
[4] Z. Liu, A. Bahety, S. Song. “REFLECT: Summarizing Robot Experiences for Failure Explanation and Correction.” CoRL 2023, PMLR 229:3468–3484.
[5] J. Duan et al. “AHA: A Vision-Language-Model for Detecting and Reasoning Over Failures in Robotic Manipulation.” ICLR 2025.
[6] P. Pacaud, R. Garcia, S. Chen. “Guardian: Detecting Robotic Planning and Execution Errors with Vision-Language Models.” CoRL 2025 Workshop.
[7] Kottom, Gharib, Humml, Stefan-Zavala. “Augmented Reality and Vision-Language Models to Guide Humans Across Manual Tasks.” SIGGRAPH 2025 Immersive Pavilion.

## 2. Design Robot Control Architecture
### 2.1 Robot Goals and System-Focused Problem
#### Interaction Design Motivation
#### Assessment Through Objective Measures

### 2.2 Architecture Design
#### Inputs
#### Outputs
#### Modules
#### Architecture Type and Justification
#### Architecture Figure

## 3. HRI Experiment Design

### 3.1 Study Participants
#### Number of Participants
#### Sampling Method

### 3.2 Experimental Design
#### Independent Variables
#### Dependent Variables
#### Study Design
#### Number of Trials
#### How the Comparison Addresses the RQs / Hypotheses
#### Evaluation Workflow Figure

### 3.3 Study Task
#### End-to-End HRI Task
#### Level of Autonomy and Human Intervention

### 3.4 Data Collection and Analysis
#### Data Sources and Types
#### Number of Observations
#### Analysis Methods

### 3.5 Metrics
#### Objective Measures

## 4. Results

### Outcomes by Dependent Variable
### Best and Worst Conditions
### Patterns Across Conditions

## 5. Discussion

### Key Findings 
#### Theme 1
#### Theme 2
### Implications for RQs / Hypotheses
### Comparison with Prior Work
### Contributions and Implications

## 6. Limitations and Future Work

### Limitations
### Future Work

## 8. Code

### Setup
### Running the Demos

## References
