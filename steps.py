"""The LEGO build: one entry per step of the instructions.
"""

BRICK_NAMES = {  # how the robot says each piece out loud
    "BLUE_2x4": "blue two by four brick",
    "BLACK_ANGLE_1x2": "black one by two angle piece",
    "GREEN_CIRCLE": "green round piece",
    "RED_1x2": "red one by two brick",
    "BLUE_1x4": "blue one by four brick",
    "BLUE_1x1": "blue one by one brick",
}

STEPS = [
    {"brick": "BLUE_1x4", "text": "Place the blue 1x4 brick on the back 4 studs of the green circle base piece."},
    {"brick": "BLUE_1x1", "text": "Put two of the blue 1x1 brick in front of the first and last stud of the blue 1x4."},
    {"brick": "RED_1x2", "text": "Put the red 1x2 brick in between the two blue 1x1 bricks."},
    {"brick": "BLUE_2x4", "text": "Put the blue 2x4 on top of the pieces you just placed."},
    {"brick": "BLACK_ANGLE_1x2", "text": "Put the black 1x2 angle piece on top, horizontally on the blue 2x4 brick, with the slope facing outward."},
    {"brick": "BLACK_ANGLE_1x2", "text": "Put another black 1x2 angle piece on top, horizontally on the blue 2x4 brick, with the slope facing outward, in front of the previous angled piece."},
    {"brick": "BLACK_ANGLE_1x2", "text": "Put another black 1x2 angle piece on top, horizontally on the blue 2x4 brick, with the slope facing outward, this time on the other side of the two angled bricks."},
    {"brick": "BLACK_ANGLE_1x2", "text": "Put the final black 1x2 angle piece on top, horizontally on the blue 2x4 brick, with the slope facing outward, in front of the previous angled brick to complete the roof."},
]

for number, step in enumerate(STEPS, start=1):
    step["number"] = number
    step["reference"] = f"reference_steps/step_{number}.jpg"
