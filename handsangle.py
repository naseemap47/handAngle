import cv2
from cvzone.HandTrackingModule import HandDetector
import math

cap = cv2.VideoCapture(2)
detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    success, img = cap.read()
    # hands, img = detector.findHands(img)  # With Draw
    hands = detector.findHands(img, draw=False)  # No Draw

    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmarks points
        # bbox1 = hand1["bbox"]  # Bounding Box info x,y,w,h
        centerPoint1 = hand1["center"]  # center of the hand cx,cy
        handType1 = hand1["type"]  # Hand Type Left or Right

        cv2.circle(
            img, centerPoint1, 5,
            (0, 255, 0), cv2.FILLED
        )

        # print(len(lmList1),lmList1)
        # print(bbox1)
        # print(centerPoint1)
        # fingers1 = detector.fingersUp(hand1)
        # length, info, img = detector.findDistance(lmList1[8], lmList1[12], img) # with draw
        # length, info = detector.findDistance(lmList1[8], lmList1[12])  # no draw

        if len(hands) == 2:
            hand2 = hands[1]
            lmList2 = hand2["lmList"]  # List of 21 Landmarks points
            # bbox2 = hand2["bbox"]  # Bounding Box info x,y,w,h
            centerPoint2 = hand2["center"]  # center of the hand cx,cy
            handType2 = hand2["type"]  # Hand Type Left or Right

            # Ref-point
            x = centerPoint2[0]
            y_ref = centerPoint2[1] - 30
            cv2.line(
                img, centerPoint2,
                (x, y_ref),
                (0, 255, 255), 2
            )

            cv2.circle(
                img, centerPoint2, 5,
                (0, 255, 0), cv2.FILLED
            )
            cv2.circle(
                img, (x, y_ref), 5,
                (0, 0, 255), cv2.FILLED
            )
            cv2.line(
                img, centerPoint1, centerPoint2,
                (0, 255, 255), 2
            )
            # fingers2 = detector.fingersUp(hand2)
            # print(fingers1, fingers2)
            # length, info, img = detector.findDistance(lmList1[8], lmList2[8], img) # with draw
            # length, info, img = detector.findDistance(centerPoint1, centerPoint2, img)  # with draw

            # Angle
            angle = math.degrees(
                math.atan2(y_ref - centerPoint2[1], x - centerPoint2[0]) -
                math.atan2(centerPoint1[1] - centerPoint2[1], centerPoint1[0] - centerPoint2[0])
            )
            print(abs(angle))
            cv2.putText(
                img, f'Angle:{int(abs(angle))}',
                (50, 50), cv2.FONT_HERSHEY_PLAIN,
                2, (0, 255, 0), 2
            )

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
