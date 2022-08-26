import cv2
import mediapipe as mp
import os


cap = cv2.VideoCapture(2)
mp_hand = mp.solutions.hands
hand = mp_hand.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Ref-points
ref1 = (250, 400)
ref2 = (20, 180)
ref3 = (250, 180)

while True:
    success, img = cap.read()
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
        img, (ref3[0]-80, ref3[1]-80),
        (ref3[0]+80, ref3[1]+80), (0, 255, 0), 3
    )

    # Process Ending Area
    cv2.rectangle(
        img, (ref2[0]-20, ref2[1]-20),
        (ref2[0]+20, ref2[1]+20), (255, 255, 0), 3
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
            if (ref3[0]-80) < line_points[0][0] < (ref3[0]+80) and (ref3[1]-80) < line_points[0][1] < (ref3[1]+80):
                cv2.putText(
                    img, 'Work Started!!', (40, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3
                )

            if (ref2[0]-20) < line_points[0][0] < (ref2[0]+20):
                if (ref2[1]-20) < line_points[0][1] < (ref2[1]+20):
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

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
