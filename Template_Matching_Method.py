import cv2
import numpy as np

#<============================TEMPLATES============================>
shape = "unrecognised"
shape_template = cv2.imread('shape.png',0)
redSign_template = cv2.imread('redSign.png',0)
yellowSign_template = cv2.imread('yellowSign.png',0)
blueSign_template = cv2.imread('blueSign.png',0)
greenSign_template = cv2.imread('greenSign.png',0)
goal_template = cv2.imread('goal.png', 0)
slope_template = cv2.imread('slope.png', 0)
traffic_template = cv2.imread('traffic.png', 0)
distance_template = cv2.imread('distance.png', 0)
#<============================VARIABLES============================>
# R_L = (0, 0, 49) #color values for masking NOTES: use hsv values better
# R_H = (7, 62, 101)
# Y_L = (0, 98, 141)
# Y_H = (63, 174, 212)
# B_L = (0, 255, 0)
# B_H = (0, 255, 0)
# G_L = (0, 255, 0)
# G_H = (0, 255, 0)
R_L = (0, 208, 69) #hsv values for masking
R_H = (10, 255, 255)
Y_L = (0, 239, 84)
Y_H = (179, 255, 219)
B_L = (91, 176, 37)
B_H = (130, 255, 255)
G_L = (35, 49, 5)
G_H = (111, 255, 42)
threshold = 0.5 #threshold for template matching
w, h = shape_template.shape[::-1]
kernel = np.ones((2,2), np.uint8)   #kernel size for morphology
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#fourcc = cv2.VideoWriter_fourcc(*'XVID') #record video
sign_detected = 0
#<============================FUNCTIONS============================>
def checkTemplate(template): #match template and draw rectangle on it
    global pt  #variable in function is local, so need to declare global to use outside
    global crop
    global hsv_crop
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
        hsv_crop = hsv[int(pt[1] + 0.1*h):int(pt[1] + 0.9*h), int(pt[0] + 0.1*w):int(pt[0] + 0.9*w)]
    if loc[1].size > 0 :
        return 1
def findColourArea(L, H): #find colour in colour signboard to differentiate them
    Thresh = cv2.inRange(hsv_crop, L, H)
    contour, hierarchy = cv2.findContours(Thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        c = max(contour, key=cv2.contourArea)
        area = cv2.contourArea(c)
        ((circle_x, circle_y), radius) = cv2.minEnclosingCircle(c)
        cv2.circle(frame, (int(circle_x+ pt[0] + 0.1*w ), int(circle_y + pt[1] + 0.1*h)), int(radius),
                                 (255, 255, 255), 3)
        return area
    except:
        return 0
def findMoments(c):
    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    return cX, cY
def testprint():
    print(shape)
    print("Peri: ", peri)
    print("Area: ", area)
#<============================MAIN_PROGRAM============================>

while True:
    if not sign_detected:
        event = "Looking for traffic sign..."
    sign_detected = 0
    tcount = 0
    scount = 0
    ccount = 0
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # --------------Colour Signboard--------------
    if checkTemplate(redSign_template) or checkTemplate(yellowSign_template) or checkTemplate(blueSign_template) or checkTemplate(greenSign_template):
        sign_detected = 1
        greenArea = findColourArea(G_L, G_H)
        redArea = findColourArea(R_L, R_H)
        yellowArea = findColourArea(Y_L, Y_H)
        blueArea = findColourArea(B_L, B_H)
        if redArea > yellowArea and redArea > blueArea and redArea > greenArea:
            event = "follow red line"

        elif yellowArea > redArea and yellowArea > blueArea and yellowArea > greenArea:
            event = "follow yellow line"

        elif blueArea > yellowArea and blueArea > redArea and blueArea > greenArea:
            event = "follow blue line"

        elif greenArea > redArea and greenArea > yellowArea and greenArea > blueArea:
            event = "follow green line"

    # --------------Shape Detection and Counting--------------
    elif checkTemplate(shape_template):
        sign_detected = 1
        print("Detecting shape(s)....")
        pinkThresh = cv2.inRange(crop, (0, 172, 14), (255, 255, 99))
        blurred = cv2.GaussianBlur(pinkThresh, (3, 3), 0)
        erode = cv2.erode(blurred, kernel, iterations=4)
        # dilate = cv2.dilate(erode, kernel, iterations=1)
        contour, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # contour = contour[0] if len(contour) == 2 else contour[1]
        cv2.imshow('Masking to find shape', erode)
        for c in contour:
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True)
            if area > 75 and area < 2300 and peri < 250 and peri >40:
                cX, cY = findMoments(c)
                    # draw the contour and center of the shape on the image
                approx = cv2.approxPolyDP(c, 0.04 * peri, True)
                if len(approx) == 3:
                    tcount+=1
                    shape = "triangle"

                elif len(approx) == 4:
                    scount+=1
                    shape = "square"

                elif len(approx) > 4 and area > 150:
                    ccount+=1
                    shape = "circle"

                #cv2.drawContours(crop, [c], -1, (255, 0, 0), 2)
                cv2.circle(crop, (cX, cY), 7, (255, 255, 255), -1)
                cv2.putText(crop, shape, (cX - 20, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.imshow('shape', crop)
        if (tcount or scount or ccount):
            print("Total ", tcount, " triangle(s)")
            print("Total ", scount, " square(s)")
            print("Total ", ccount, " circle(s)")
    # --------------Measure slope angle--------------
    elif checkTemplate(slope_template):
        sign_detected = 1
        event = "Measure Slope Angle"
    # --------------Traffic Light--------------
    elif checkTemplate(traffic_template):
        sign_detected = 1
        event = "Traffic light!"
    # --------------Soccer mode--------------
    elif checkTemplate(goal_template):
        sign_detected = 1
        event = "Football!"
    # --------------Measure Distance--------------
    elif checkTemplate(distance_template):
        sign_detected = 1
        event = "Measure distance"
    cv2.putText(frame, event, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()


