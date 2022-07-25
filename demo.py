import cv2
import streamlit as st
import mediapipe as mp
import math
from findPoints import find_points


st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(2)

mp_hand = mp.solutions.hands
hand = mp_hand.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Ref-points
ref1 = (200, 400)
ref2 = (20, 180)
ref3 = (250, 180)
line_points = []
x_list = []
y_list = []

while run:
    success, img = camera.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hand.process(img_rgb)

    # Ref-Line
    cv2.line(
        img, ref3, ref1,
        (0, 255, 255), 2
    )
    cv2.line(
        img, ref3, ref2,
        (0, 255, 255), 2
    )
    cv2.circle(
        img, ref1,
        5, (0, 0, 255), cv2.FILLED
    )
    cv2.circle(
        img, ref2,
        5, (0, 0, 255), cv2.FILLED
    )
    cv2.circle(
        img, ref3,
        5, (0, 0, 255), cv2.FILLED
    )

    if results.multi_hand_landmarks:
        landmaks_list = []
        for handType, hand_lm in zip(results.multi_handedness, results.multi_hand_landmarks):
            for id, lm in enumerate(hand_lm.landmark):
                h, w, c = img.shape
                x, y = int(lm.x * w), int(lm.y * h)
                if id == 8:
                    landmaks_list.append([handType.classification[0].label, x, y])
                    if landmaks_list[0][0] == 'Left':
                        r_x = landmaks_list[0][1]
                        r_y = landmaks_list[0][2]
                        line_points.append([r_x, r_y])
        # print(line_points)
        if len(line_points) > 2:
            for point in line_points:
                cv2.line(
                    img, point,
                    point, (0, 255, 0), 5
                )

                

    img_rgb_out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    FRAME_WINDOW.image(img_rgb_out)
    # if predict:
    if len(line_points) > 10:
        p1, p2, p3, p4 = find_points(line_points)
        cv2.circle(
            img, p1, 5, (255, 0, 0), cv2.FILLED
        )
        cv2.circle(
            img, p2, 5, (0, 255, 0), cv2.FILLED
        )
        # cv2.circle(
        #     img, p3, 5, (0, 0, 255), cv2.FILLED
        # )
        cv2.circle(
            img, p4, 5, (255, 0, 255), cv2.FILLED
        )

        # Line
        cv2.line(
            img, p1, p2,
            (0, 255, 255), 2
        )
        cv2.line(
            img, p2, p4,
            (0, 255, 255), 2
        )

        # Angle
        angle = math.degrees(
            math.atan2(p4[1] - p2[1], p4[0] - p2[0]) -
            math.atan2(p1[1] - p2[1], p1[0] - p2[0])
        )
        # print(angle-180)
        cv2.putText(
            img, f'Angle:{int(angle-180)}',
            (50, 50), cv2.FONT_HERSHEY_PLAIN,
            2, (0, 255, 0), 2
        )

    img_rgb_out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    FRAME_WINDOW.image(img_rgb_out)


else:
    st.markdown('Stopped!!')

