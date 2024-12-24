import cv2
import os

def video_segment():
    vidcap = cv2.VideoCapture('./static/video/file.mp4')
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    count = 0
    flag = 1
    while True:
        success, image = vidcap.read()
        if not success:
            break
        if count % int(fps) == 0:
            cv2.imwrite(f"./static/img/{flag}.jpg", image)
            # print(flag)
            flag += 1
        count += 1
    vidcap.release()
    os.remove('./static/video/file.mp4')

