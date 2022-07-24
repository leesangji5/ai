import cv2
import numpy as np

model = 'data/res10_300x300_ssd_iter_140000_fp16.caffemodel'
config = 'data/deploy2.prototxt'

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Camera open failed!')
    exit()

net = cv2.dnn.readNet(model, config)

if net.empty():
    print('Net open failed!')
    exit()

mos = 5

while True:
    _, frame = cap.read()
    if frame is None:
        break

    blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123))
    net.setInput(blob)
    detect = net.forward()

    detect = detect[0, 0, :, :]
    (h, w) = frame.shape[:2]

    for i in range(detect.shape[0]):
        confidence = detect[i, 2]
        if confidence < 0.5:
            break

        x1 = int(detect[i, 3] * w*0.9)
        y1 = int(detect[i, 4] * h*0.75)
        x2 = int(detect[i, 5] * w*1.1)
        y2 = int(detect[i, 6] * h*1.1)
        x = int(x2-x1)
        y = int(y2-y1)
        mx = int(x/mos)
        my = int(y/mos)
        face = frame[y1:y2, x1:x2]
        face = cv2.resize(face, (mx, my))
        face = cv2.resize(face, (x, y))
        frame[y1:y2, x1:x2] = face

        label = 'Face: %4.3f' % confidence
        cv2.putText(frame, label, (x1, y1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()