import numpy as np
import mediapipe as mp
from collections import deque
import streamlit as st
from PIL import Image
import cv2

# Initialize the Streamlit app
st.title("Hand-Tracking Paint Application")

# Sidebar for options
st.sidebar.title("Controls")
clear_button = st.sidebar.button("Clear Canvas")

# Setting up canvas parameters
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

blue_index = green_index = red_index = yellow_index = 0

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# Create a blank white canvas
paintWindow = np.ones((471, 636, 3), dtype=np.uint8) * 255
paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), (255, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), (0, 255, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), (0, 0, 255), 2)
paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), (0, 255, 255), 2)

# Labels for the buttons
cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Streamlit function for live frame display
frame_window = st.image([])

# Allow users to upload an image or use the camera
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Camera input from Streamlit
camera_image = st.camera_input("Take a picture")

if camera_image:
    # Process the camera input
    frame = np.array(camera_image)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
else:
    if uploaded_file:
        # Process the uploaded image
        image = Image.open(uploaded_file)
        frame = np.array(image)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        st.warning("Please upload an image or use the camera.")
        st.stop()

# Add the canvas and color buttons to the frame
frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)
frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 2)
frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)
frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 255), 2)

# Get hand landmark prediction
result = hands.process(frame_rgb)

if result.multi_hand_landmarks:
    landmarks = []
    for handslms in result.multi_hand_landmarks:
        for lm in handslms.landmark:
            lmx = int(lm.x * frame.shape[1])
            lmy = int(lm.y * frame.shape[0])
            landmarks.append([lmx, lmy])

        mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
    fore_finger = (landmarks[8][0], landmarks[8][1])
    center = fore_finger
    thumb = (landmarks[4][0], landmarks[4][1])

    # Gesture detection for drawing
    if thumb[1] - center[1] < 30:
        bpoints.append(deque(maxlen=512))
        blue_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        rpoints.append(deque(maxlen=512))
        red_index += 1
        ypoints.append(deque(maxlen=512))
        yellow_index += 1
    elif center[1] <= 65:
        if 40 <= center[0] <= 140:
            bpoints = [deque(maxlen=512)]
            gpoints = [deque(maxlen=512)]
            rpoints = [deque(maxlen=512)]
            ypoints = [deque(maxlen=512)]
            blue_index = green_index = red_index = yellow_index = 0
            paintWindow[67:, :, :] = 255  # Clear the canvas
        elif 160 <= center[0] <= 255:
            colorIndex = 0  # Blue
        elif 275 <= center[0] <= 370:
            colorIndex = 1  # Green
        elif 390 <= center[0] <= 485:
            colorIndex = 2  # Red
        elif 505 <= center[0] <= 600:
            colorIndex = 3  # Yellow
    else:
        if colorIndex == 0:
            bpoints[blue_index].appendleft(center)
        elif colorIndex == 1:
            gpoints[green_index].appendleft(center)
        elif colorIndex == 2:
            rpoints[red_index].appendleft(center)
        elif colorIndex == 3:
            ypoints[yellow_index].appendleft(center)

points = [bpoints, gpoints, rpoints, ypoints]
for i in range(len(points)):
    for j in range(len(points[i])):
        for k in range(1, len(points[i][j])):
            if points[i][j][k - 1] is None or points[i][j][k] is None:
                continue
            cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
            cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

frame_window.image(frame)  # Display the live frame in Streamlit

if clear_button:
    bpoints = [deque(maxlen=1024)]
    gpoints = [deque(maxlen=1024)]
    rpoints = [deque(maxlen=1024)]
    ypoints = [deque(maxlen=1024)]
    paintWindow[67:, :, :] = 255
