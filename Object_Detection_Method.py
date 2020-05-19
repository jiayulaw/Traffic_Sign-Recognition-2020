import numpy as np
import cv2
import pickle

#<=====================SETUP=====================>
frameWidth= 640         # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.8        # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)#'RecordTemplate002.h264'
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
kernel = np.ones((2,2), np.uint8)   #kernel size for morphology
R_L = (0, 208, 69) #hsv values for masking
R_H = (10, 255, 255)
Y_L = (0, 239, 84)
Y_H = (179, 255, 219)
B_L = (91, 176, 37)
B_H = (130, 255, 255)
G_L = (35, 49, 5)
G_H = (111, 255, 42)
#<=====================IMPORT TRAINED MODEL(s)=====================>
pickle_in=open("Sign_Recognizer_CNN.p","rb")  ## rb = READ BYTE
model=pickle.load(pickle_in)
sign_cascade = cv2.CascadeClassifier('haar_traffic_sign_detector.xml')
#<=====================FUNCTIONS=====================>
def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img
def getClassName(classNo):
    if   classNo == 0: return 'Short Cut - Yellow'
    elif classNo == 1: return 'Short Cut - Red'
    elif classNo == 2: return 'Short Cut - Blue'
    elif classNo == 3: return 'Short Cut - Green'
    elif classNo == 4: return 'Distance Measurement Event'
    elif classNo == 5: return 'Traffic Light'
    elif classNo == 6: return 'Football!'
    elif classNo == 7: return 'Incline Measurement'
    elif classNo == 8: return 'Push Button'
    elif classNo == 9: return 'Wall Of Shapes'
def findMoments(c):
    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    return cX, cY
def findColourArea(L, H): #find colour in colour signboard to differentiate them
    Thresh = cv2.inRange(hsv_crop, L, H)
    contour, hierarchy = cv2.findContours(Thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.imshow('Masked', cv2.resize(Thresh, (320,240)))
    try:
        c = max(contour, key=cv2.contourArea)
        area = cv2.contourArea(c)
        ((circle_x, circle_y), radius) = cv2.minEnclosingCircle(c)
        #Draw a minimum enclosing circle around the colour
        #cv2.drawContours(hsv_crop, [c], -1, (0, 255, 0), 1)
        cv2.circle(frame, (int(circle_x + x + 0.6*w), int(circle_y + y + 0.5*h)), int(radius),
                   (255, 255, 255), 2)
        return area
    except:
        return 0
#<=====================MAIN PROGRAM=====================>
while True:
    text = "Detecting Sign..."
    tcount = 0
    scount = 0
    ccount = 0
    pcount = 0
    success, frame = cap.read()
    cv2.rectangle(frame, (3, 0), (300, 85), (255, 255, 255), cv2.FILLED)
    cv2.rectangle(frame, (3, 0), (300, 85), (0, 255, 255), 2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    signs = sign_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
    for (x, y, w, h) in signs:
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, int(y - 0.1*h)), (x + w, y), (0, 255, 0), cv2.FILLED)
        imgOriginal = frame[y: y+h, x: x+w]
        # PROCESS IMAGE
        img = np.asarray(imgOriginal)
        img = cv2.resize(img, (32, 32))
        img = preprocessing(img)
        cv2.imshow("Processed Image", img)
        img = img.reshape(1, 32, 32, 1)

        # PREDICT IMAGE
        predictions = model.predict(img)
        classIndex = model.predict_classes(img)
        probabilityValue = np.amax(predictions)

        if probabilityValue > threshold:
            obj = getClassName(classIndex)
            #================Some Extra Task(s)================
            # --------------Shape Detection and Counting--------------
            if obj == 'Wall Of Shapes':
                print("Detecting shape(s)....")
                crop = gray[int(y+0.1*h): int(y+h-0.13*h), int(x+0.1*w): int(x+w-0.15*w)]
                #ret, thresh = cv2.threshold(crop, 51, 79, cv2.THRESH_BINARY_INV)
                # blurred = cv2.GaussianBlur(crop, (1, 1), 0)
                blurred = cv2.bilateralFilter(crop, 5, 150,150)
                # erode = cv2.erode(blurred, (1,1), iterations=1)
                dilate = cv2.dilate(blurred, (1,1), iterations=1)
                # erode = cv2.erode(dilate, (1, 1), iterations=2)
                ret, otsu = cv2.threshold(dilate, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                cv2.imshow('masked', cv2.resize(otsu, (320,240)))
                contour, hierarchy = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                try:
                    biggest_shape = max(contour, key=cv2.contourArea)
                    biggest_shape_area = cv2.contourArea(biggest_shape)
                    for c in contour:
                        area = cv2.contourArea(c)
                        peri = cv2.arcLength(c, True)
                        shape = "unknown"
                        if area > 0.06 * biggest_shape_area:  # if the area is not smaller than certain
                            cX, cY = findMoments(c)  # percent of the biggest shape
                            # draw the contour and center of the shape on the image
                            approx = cv2.approxPolyDP(c, 0.045 * peri, True)
                            if len(approx) == 3:
                                tcount += 1
                                shape = "triangle"

                            elif len(approx) == 4:
                                scount += 1
                                shape = "quadrilateral"

                            # elif len(approx) == 5:
                            #     pcount += 1
                            #     shape = "pentagon"

                            elif len(approx) > 4:
                                ccount += 1
                                shape = "circle"

                            cv2.drawContours(frame, [c], -1, (255, 0, 0), 2,
                                             offset=(int(0.1 * w + x), int(0.1 * h + y)))
                            cv2.circle(frame, (int(cX + x + 0.1 * w), int(cY + y + 0.1 * h)), 2, (0, 0, 255), -1)
                            cv2.putText(frame, shape, (int(cX + x - 20 + 0.1 * w), int(cY + y - 20 + 0.1 * h)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1*w/640, (255, 255, 255), 2)

                    if (tcount or scount or ccount or pcount):
                        print("Total ", tcount, " triangle(s)")
                        print("Total ", scount, " quadrilateral(s)")
                        print("Total ", ccount, " circle(s)")
                        print("Total ", pcount, " pentagon(s)")
                        cv2.putText(frame, str(tcount) + " triangle(s),    " + str(scount) + " quadrilateral(s)",
                                    (5, 45), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(frame, str(ccount) + " circle(s)", (5, 65),
                                    font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                except:
                    print("No shape detected")

            #--------------Confirming Colour Short-cut--------------
            if obj == 'Short Cut - Red' or obj == 'Short Cut - Yellow' or obj == 'Short Cut - Blue' or obj == 'Short Cut - Green':
                hsv_crop = hsv[int(y+0.5*h): int(y+h-0.13*h), int(x+0.6*w): int(x+w-0.15*w)]
                greenArea = findColourArea(G_L, G_H)
                redArea = findColourArea(R_L, R_H)
                yellowArea = findColourArea(Y_L, Y_H)
                blueArea = findColourArea(B_L, B_H)
                if redArea > (yellowArea + blueArea + greenArea) :
                    print("follow red line")
                    obj = 'Short Cut - Red'
                elif yellowArea > (redArea + blueArea + greenArea):
                    print("follow yellow line")
                    obj = 'Short Cut - Yellow'
                elif blueArea > (yellowArea + redArea + greenArea):
                    print("follow blue line")
                    obj = 'Short Cut - Blue'
                elif greenArea > (yellowArea + blueArea + redArea):
                    print("follow green line")
                    obj = 'Short Cut - Green'

            text = str(classIndex) + " " + obj + " " + str(round(probabilityValue * 100, 2)) + "%"
        #--------------Finally, print out the predictions--------------
        cv2.putText(frame, text, (x, int(y - 0.03*h)), font, 1.3*w/640, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, text, (5, 25), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)


    cv2.imshow("Result", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

