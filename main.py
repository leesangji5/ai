import cv2
import numpy as np
import dlib
import sys

# mosaic settings
model = 'data/res10_300x300_ssd_iter_140000_fp16.caffemodel'
config = 'data/deploy2.prototxt'

net = cv2.dnn.readNet(model, config)

mos = 5

# imoji settings
scaler = 1

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

overlay = cv2.imread('samples/ryan_transparent.png', cv2.IMREAD_UNCHANGED)

def mosaic(frame, mos):
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

    return frame

def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    bg_img = background_img.copy()
    if bg_img.shape[2] == 3:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    b, g, r, a = cv2.split(img_to_overlay_t)

    mask = cv2.medianBlur(a, 5)

    h, w, _ = img_to_overlay_t.shape
    roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

    bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)

    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

    return bg_img

def imoji(frame, scaler):
    face_roi = []
    face_sizes = []

    frame = cv2.resize(frame, (int(frame.shape[1] * scaler), int(frame.shape[0] * scaler)))

    if len(face_roi) == 0:
        faces = detector(frame, 1)
    else:
        roi_img = frame[face_roi[0]:face_roi[1], face_roi[2]:face_roi[3]]
        faces = detector(roi_img)
    
    for face in faces:
        if len(face_roi) == 0:
            dlib_shape = predictor(frame, face)
            shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])
        else:
            dlib_shape = predictor(roi_img, face)
            shape_2d = np.array([[p.x + face_roi[2], p.y + face_roi[0]] for p in dlib_shape.parts()])

        for s in shape_2d:
            cv2.circle(frame, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)

        min_coords = np.min(shape_2d, axis=0)
        max_coords = np.max(shape_2d, axis=0)

        cv2.circle(frame, center=tuple(min_coords), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(frame, center=tuple(max_coords), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

        face_size = max(max_coords - min_coords)
        face_sizes.append(face_size)
        if len(face_sizes) > 10:
            del face_sizes[0]
        mean_face_size = int(np.mean(face_sizes) * 1.8)

        face_roi = np.array([int(min_coords[1] - face_size / 2), int(max_coords[1] + face_size / 2), int(min_coords[0] - face_size / 2), int(max_coords[0] + face_size / 2)])
        face_roi = np.clip(face_roi, 0, 10000)

        frame = overlay_transparent(frame, overlay, center_x + 8, center_y - 25, overlay_size=(mean_face_size, mean_face_size))

    return frame

cap = cv2.VideoCapture(0)

run = True

while run:
    _, frame = cap.read()
    ori = frame.copy()
    if frame is None:
        break

    frame = imoji(ori, scaler)

    frame = mosaic(ori, mos)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:
        run = False
