import numpy as np


def find_points(approx):
    x_list = []
    y_list = []
    for point in approx:
        x_list.append(point[0])
        y_list.append(point[1])

    # Edge points of patches
    y_avg = np.average(y_list)

    # TOP
    # Below Average
    y_belowAvg = []
    x_belowAvg = []
    for i in range(len(y_list)):
        if y_list[i] < y_avg:
            y_belowAvg.append(y_list[i])
            x_belowAvg.append(x_list[i])

    # points 1
    x1_id = np.argmin(x_belowAvg)
    e_x1 = x_belowAvg[x1_id]
    e_y1 = y_belowAvg[x1_id]

    # points 2
    x2_id = np.argmax(x_belowAvg)
    e_x2 = x_belowAvg[x2_id]
    e_y2 = y_belowAvg[x2_id]

    # BOTTOM
    # Above Average
    y_aboveAvg = []
    x_aboveAvg = []
    for i in range(len(y_list)):
        if y_list[i] > y_avg:
            y_aboveAvg.append(y_list[i])
            x_aboveAvg.append(x_list[i])

    # points 3
    x3_id = np.argmin(x_aboveAvg)
    e_x3 = x_aboveAvg[x3_id]
    e_y3 = y_aboveAvg[x3_id]

    # points 4
    x4_id = np.argmax(x_aboveAvg)
    e_x4 = x_aboveAvg[x4_id]
    e_y4 = y_aboveAvg[x4_id]

    # Points
    p1 = (e_x1, e_y1)
    p2 = (e_x2, e_y2)
    p3 = (e_x3, e_y3)
    p4 = (e_x4, e_y4)
    return p1, p2, p3, p4

