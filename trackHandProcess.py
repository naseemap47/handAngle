import cv2
import mediapipe as mp
import os
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("--source", type=str, required=True,
                help="Web-cam port or path to video")
ap.add_argument("--source_width", type=int, required=True,
                help="Width of source (web-cam or video)")
ap.add_argument("--source_height", type=int,
                help="Height of source (web-cam or video)")
ap.add_argument("--save", action='store_true',
                help="Save video")

args = vars(ap.parse_args())
source = args["source"]
source_width = args["source_width"]
source_height = args["source_height"]
save = args['save']

if source.isnumeric():
    source = int(source)

cap = cv2.VideoCapture(source)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, source_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, source_height)

# Write Video
if save:    
    result = cv2.VideoWriter('output.avi',
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            10, (source_width, source_height))

mp_hand = mp.solutions.hands
hand = mp_hand.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Ref-points
ref1 = (int(source_width*0.46875), int(source_height*0.8333))
ref2 = (int(source_width*0.109375), int(source_height*0.375))
ref3 = (int(source_width*0.46875), int(source_height*0.375))

working_area_size = 130
end_area_size = 35

while True:
    success, img = cap.read()
    if not success:
        break
    
    img = cv2.resize(img, (source_width, source_height))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hand.process(img_rgb)

    # Ref-Line
    cv2.line(
        img, ref2, ref1,
        (0, 255, 255), 2
    )
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

                if id == 8 or id == 4:
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

    # Write Video
    if save:
        result.write(img)

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if save:
    result.release()
cv2.destroyAllWindows()
