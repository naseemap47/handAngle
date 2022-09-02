import streamlit as st
import cv2
import mediapipe as mp
import os


st.title('Hand Process Checker')
st.sidebar.title('Settings')
FRAME_WINDOW = st.image([])
# Mediapipe
mp_hand = mp.solutions.hands
hand = mp_hand.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Webcam Channels
options = st.sidebar.radio('Options:', ('Webcam', 'Video'))
width = st.sidebar.number_input('Width:', value=640, min_value=100)
height = st.sidebar.number_input('Height:', value=480, min_value=100)

if options == 'Webcam':
    cam_options = st.sidebar.selectbox('Webcam Channel',
                                        ('Select Channel', '0', '1', '2', '3'))

    if not cam_options == 'Select Channel':
        cap = cv2.VideoCapture(int(cam_options))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))


if options == 'Video':
    video_option = st.sidebar.text_input('Path to Video:')
    if st.sidebar.checkbox('Set Video Path'):
        cap = cv2.VideoCapture(video_option)

# Ref-points
# ref1 = (int(width*0.46875), int(height*0.8333))
ref1_x = st.sidebar.slider('ref1_x:', 0, 1000, int(width*0.46875))
ref1_y = st.sidebar.slider('ref1_y:', 0, 1000, int(height*0.8333))
ref1 = (int(ref1_x), int(ref1_y))
# ref2 = (int(width*0.109375), int(height*0.375))
ref2_x = st.sidebar.slider('ref2_x:', 0, 1000, int(width*0.109375))
ref2_y = st.sidebar.slider('ref2_y:', 0, 1000, int(height*0.375))
ref2 = (int(ref2_x), int(ref2_y))
# ref3 = (int(width*0.46875), int(height*0.375))
ref3_x = st.sidebar.slider('ref3_x:', 0, 1000, int(width*0.46875))
ref3_y = st.sidebar.slider('ref3_y:', 0, 1000, int(height*0.375))
ref3 = (int(ref3_x), int(ref3_y))

# Area Size
working_area_size = st.sidebar.slider('Working Area Size:', 0, 200, 80)
end_area_size = st.sidebar.slider('End Area Size:', 0, 100, 30)

if st.checkbox('Start Process'):    
    while True:
        success, img = cap.read()
        if not success:
            st.warning('[INFO] Camera NOT Working!!')
            break
        img = cv2.resize(img, (width, height))
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
            5, (0, 255, 0), cv2.FILLED
        )
        cv2.circle(
            img, ref3,
            5, (255, 0, 255), cv2.FILLED
        )

        # Working Area
        cv2.rectangle(
            img, (ref3[0]-working_area_size, ref3[1]-working_area_size),
            (ref3[0]+working_area_size, ref3[1]+working_area_size), (0, 255, 0), 3
        )

        # Process Ending Area
        cv2.rectangle(
            img, (ref2[0]-end_area_size, ref2[1]-end_area_size),
            (ref2[0]+end_area_size, ref2[1]+end_area_size), (255, 255, 0), 3
        )

        # Flip Webcam feed
        img = cv2.flip(img, -1)

        if results.multi_hand_landmarks:
            landmaks_list = []
            line_points = []
            for handType, hand_lm in zip(results.multi_handedness, results.multi_hand_landmarks):
                for id, lm in enumerate(hand_lm.landmark):
                    h, w, c = img.shape
                    x, y = int(lm.x * w), int(lm.y * h)

                    if id == 8:
                        landmaks_list.append(
                            [handType.classification[0].label, x, y])
                        if landmaks_list[0][0] == 'Left':
                            r_x = landmaks_list[0][1]
                            r_y = landmaks_list[0][2]
                            line_points.append([r_x, r_y])

            # print(line_points)
            if len(line_points) > 0:

                # Working Area
                if (ref3[0]-working_area_size) < line_points[0][0] < (ref3[0]+working_area_size) and (ref3[1]-working_area_size) < line_points[0][1] < (ref3[1]+working_area_size):
                    cv2.putText(
                        img, 'Work Started!!', (40, 50),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3
                    )

                if (ref2[0]-end_area_size) < line_points[0][0] < (ref2[0]+end_area_size):
                    if (ref2[1]-end_area_size) < line_points[0][1] < (ref2[1]+end_area_size):
                        cv2.putText(
                            img, 'Work Ended!!', (40, 50),
                            cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3
                        )
                    else:
                        os.system('spd-say "Warning"')
                        cv2.putText(
                            img, 'Warning!!', (40, 50),
                            cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3
                        )

            else:
                cv2.putText(
                    img, 'Hand NOT Detected!!', (40, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3
                )
            FRAME_WINDOW.image(img, channels='BGR')
