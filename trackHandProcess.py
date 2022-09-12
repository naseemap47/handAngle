import cv2
import mediapipe as mp
import os
import argparse
import pyshine as ps


ap = argparse.ArgumentParser()
ap.add_argument("--source", type=str, required=True,
                help="Web-cam port or path to video")
# ap.add_argument("--source_width", type=int, required=True,
#                 help="Width of source (web-cam or video)")
# ap.add_argument("--source_height", type=int, required=True,
#                 help="Height of source (web-cam or video)")
ap.add_argument("--save", action='store_true',
                help="Save video")


args = vars(ap.parse_args())
source = args["source"]
# source_width = args["source_width"]
# source_height = args["source_height"]
save = args['save']

source_width = 1300
source_height = 650

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
# ref1 = (int(source_width*0.46875), int(source_height*0.8333))
# ref2 = (int(source_width*0.109375), int(source_height*0.375))
# ref3 = (int(source_width*0.46875), int(source_height*0.375))

ref1 = (int(source_width*0.46875), 450)
ref2 = (1000, 450)
ref3 = (int(source_width*0.46875), int(source_height*0.375))

working_area_size = 130
end_area_size = 35
total_work_done = 0
Flage = 0

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
        img, (ref1[0]-working_area_size, ref1[1]-working_area_size),
        (ref1[0]+working_area_size, ref1[1]+working_area_size), (0, 255, 0), 3
    )

    # Process Ending Area
    cv2.rectangle(
        img, (ref2[0]-end_area_size, ref2[1]-end_area_size),
        (ref2[0]+end_area_size, ref2[1]+end_area_size), (255, 255, 0), 3
    )

    # total_work_done
    # cv2.putText(
    #     img, f'Total Work Done: {total_work_done}',
    #     (40, 90), cv2.FONT_HERSHEY_PLAIN,
    #     3, (255, 0, 255), 3
    # )

    ps.putBText(
        img,f'Total Work Done: {total_work_done}',
        text_offset_x=20,text_offset_y=20,vspace=20,
        hspace=20, font_scale=1.0,
        background_RGB=(0,250,250),text_RGB=(255,255,255)
    )

    # Flip Webcam feed
    # img = cv2.flip(img, -1)

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
            if (ref1[0]-working_area_size) < line_points[0][0] < (ref1[0]+working_area_size) and (ref1[1]-working_area_size) < line_points[0][1] < (ref1[1]+working_area_size):
                # cv2.putText(
                #     img, 'Work in Progress..', (40, 50),
                #     cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3
                # )
                ps.putBText(
                    img, 'Work in Progress..',
                    text_offset_x=20, text_offset_y=83,
                    vspace=20, hspace=20, font_scale=1.0,
                    background_RGB=(0,220,0), text_RGB=(255,255,255)
                )
                
                Flage = 1

            if (ref2[0]-end_area_size) < line_points[0][0] < (ref2[0]+end_area_size):
                if (ref2[1]-end_area_size) < line_points[0][1] < (ref2[1]+end_area_size):
                    # cv2.putText(
                    #     img, 'Work Done!!', (40, 50),
                    #     cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3
                    # )
                    ps.putBText(
                        img, 'Work Done!!',
                        text_offset_x=20, text_offset_y=83,
                        vspace=20, hspace=20, font_scale=1.0,
                        background_RGB=(255,225,0), text_RGB=(255,255,255)
                    )

                    if Flage == 1:
                        total_work_done += 1
                        Flage = 0                  
                
                else:
                    os.system('spd-say "Warning"')
                    cv2.putText(
                        img, 'Warning!!', (40, 50),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3
                    )

        # else:
        #     cv2.putText(
        #         img, 'Hand NOT Detected!!', (40, 50),
        #         cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3
        #     )
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
