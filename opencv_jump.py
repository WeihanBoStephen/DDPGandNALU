import cv2
import numpy as np
def find_head(img):
    head_found = False
    trial_cnt = 0
    while(head_found == False):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        edges = cv2.Canny(closed, 200, 200)
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 2, 180 - trial_cnt * 10, param1 = 100, param2 = 65, minRadius = 12, maxRadius = 18)
        if(circles is not None):
            head_found = True
        else:
            if (trial_cnt > 5):
                return 0, 0, 0
        trial_cnt += 1
    circles = np.uint16(np.around(circles))
    xo = 0
    yo = 0
    ro = 0
    for i in circles[0, :]:
        x = i[0]
        y = i[1]
        r = i[2]
        if (r < 25 and y > 380 and y < 520 and (y < yo or yo is  0 )):
            xo=x
            yo=y
            ro=r
    return xo, yo, ro
def find_character_centre(x,y):
    xo = x
    yo = y + 75
    return (xo, yo)
def find_box_centre(img, loc_foot, index = 0):
    edges = cv2.Canny(img, 15, 70)
    points = cv2.findNonZero(edges)
    for point in points:
        x = point[0][0]
        y = point[0][1]
        if y < 250:
            continue
        if np.abs(x - loc_foot[0]) < 30:
            continue
        if loc_foot[1] - y < 30:
            continue
        if x < 50 or x > 540 - 50:
            continue
        break
    xo = x
    yo = y + 45
    if index > 33:
        yo = y + 35
    if index > 75:
        yo = y + 20
    return (xo, yo)
def get_presstime(file_name):
    flag = True
    img = cv2.imread(file_name)
    img = cv2.resize(img, (540, 960))
    i = find_head(img)
    if(i[2] == 0):
        flag = False
        rand_action = np.random.normal(-0.333720, 0.314713, 1)[0]
        press_time = (rand_action + 1)*600+200
        print("opencv can't find circle.")
        return press_time, flag
    character_centre = find_character_centre(i[0],i[1])
    box_centre = find_box_centre(img, character_centre)
    distance = np.sqrt((box_centre[0] - character_centre[0])*(box_centre[0] - character_centre[0]) + (box_centre[1] - character_centre[1])*(box_centre[1] - character_centre[1]))
    press_time = distance / 165 #365
    press_time = int(np.int(press_time*1000)+80)
    print(press_time)
    return press_time,flag
def presstime_to_action(presstime):
    action = (presstime - 200) / 600 - 1
    return action
def get_action(file_name):
    press_time,flag = get_presstime(file_name)
    return presstime_to_action(press_time)
test = get_presstime("./nalu_img/1535333631.jpg")
print(test)

