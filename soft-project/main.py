import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from methods import resize_region, distance2points, invert, scale_to_range_n_flaten
from keras import models

total_sum = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

nn = models.load_model('neuronska.h5')

i = 0

for i in range(0, 10):

    cap = cv2.VideoCapture('videos/video-' + str(i) + '.avi')

    ret, frame = cap.read()

    blank = np.copy(frame) * 0

    lines = cv2.HoughLinesP(frame[:, :, 0], 1, np.pi / 180, 90, 90, 8)

    max_length = 0

    for line in lines:
        for x1, y1, x2, y2 in line:
            curent = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if (max_length < curent):
                max_length = curent
                max = [x1, y1, x2, y2]

    line_x1 = max[0]
    line_y1 = max[1]
    line_x2 = max[2]
    line_y2 = max[3]

    cv2.line(blank, (line_x1, line_y1), (line_x2, line_y2), (255, 0, 0), 10)

    # plt.imshow(frame[:, :, 0])
    # plt.imshow(blank)
    # plt.show()
    # nasao je liniju

    #########

    # Broj frejmova
    j = 0

    all_detected = []

    while cap.isOpened():
        j = j + 1
        gotMore, frame = cap.read()

        if not gotMore:
            break

        cv2.line(frame, (line_x1, line_y1), (line_x2, line_y2), (0, 0, 255), 5)  # (slika,t1,t2,color,thickness)

        #print('linija-kordinate', line_x1, line_y1, line_x2, line_y2)
        # linija-kordinate 288 204 492 50

        # izdvoj cifrew
        low_white = np.array([160, 160, 160])
        up_white = np.array([245, 245, 245])
        maska_white = cv2.inRange(frame, low_white, up_white)

        # frame = invert(image_bin(image_gray(frame)))

        nums = cv2.bitwise_and(frame, frame, mask=maska_white)

        kernel = (5, 5)

        found_nums = cv2.GaussianBlur(cv2.cvtColor(nums, cv2.COLOR_BGR2GRAY), kernel, 0)

        #plt.imshow(found_nums)
        #plt.show()
        #Pokaze samo brojeve

        img, contours, hierarchy = cv2.findContours(found_nums.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        cords = []
        regions_array = []

        for contour in contours:

            (x, y, w, h) = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            if 1000 > area > 30 and h > 15 and w > 10:
                region = found_nums[y: y + h + 1, x:x + w + 1]
                regions_array.append([resize_region(region), (x, y, w, h)])
                cv2.rectangle(found_nums, (x, y), (x + w, y + h), (0, 255, 0), 3)

                cord = (x, y, w, h)
                cords.append(cord)

        for cord in cords:
            x, y, w, h = cord

            # uzimamo original sliku i za date kordinate isecamo okvir od vaznosti
            # menjamo velicinu isecka da bude pogodan za obradu na ulazu neuronske
            number = found_nums[y:y + h, x:x + w]
            number = resize_region(number)

            a = distance2points([line_x1, line_y1], [x, y])
            b = distance2points([line_x2, line_y2], [x, y])
            c = distance2points([line_x1, line_y1], [line_x2, line_y2])

            avg = (a + b + c) / 2
            close2line = 2 * math.sqrt(avg * (avg - a) * (avg - b) * (avg - c)) / a

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # smanjuje zamucenje
            ret, number = cv2.threshold(invert(number), 120, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            if (close2line < 5) and (line_y2 < y < line_y1 and line_x2 > x > line_x1):
                # otprilike na 4 je broj malo ispod linije

                wasAdded = False

                if len(all_detected):
                    for det in all_detected:
                        if det[0] == x or det[1] == y:
                            wasAdded = True

                        if not wasAdded:
                            if distance2points([x, y], [det[0], det[1]]) < 25 and (j - det[2]) < 25:
                                wasAdded = 1

                if not wasAdded:
                    all_detected.append((x, y, j))  # ubacivanje u listu zajedno sa brojem frejma

                    plt.imshow(number)
                    plt.imshow(frame)

                    plt.show()

                    number = scale_to_range_n_flaten(number)
                    rdyForNN = np.array([number], np.float32)

                    retFromNN = nn.predict_classes(rdyForNN)
                    total_sum[i] = total_sum[i] + retFromNN[0]

    cap.release()
os.remove("out.txt")
file = open('out.txt', 'w+')
file.write('RA 143/2015 Igor Lovric \n')
file.write('file sum')
for i in range(10):
    file.write('\n' + 'video-' + str(i) + '.avi' + '\t' + str(total_sum[i]))
file.close()
print('File saved successfully.')
