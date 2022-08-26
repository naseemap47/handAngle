import cv2
import mediapipe as mp
import math

cap = cv2.VideoCapture(2)

mp_hand = mp.solutions.hands
hand = mp_hand.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Ref-points
ref1 = (240, 400)
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
        img, (150, 80), (350, 280),
        (0, 255, 0), 3
    )

    if results.multi_hand_landmarks:
        landmaks_list = []
        line_points = []
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
    
        # Working Area
        if len(line_points)>0:
            if 150<line_points[0][0]<350 and 80<line_points[0][0]<280:
                cv2.putText(
                    img, 'Work Started!!', (40, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3
                )
        else:
            cv2.putText(
                    img, 'Hand NOT Detected!!', (40, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3
                )

    # img = cv2.flip(img,-1)
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
