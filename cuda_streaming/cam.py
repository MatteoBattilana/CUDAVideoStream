import sys
import cv2

def read_cam():

    # cap = cv2.VideoCapture("v4l2src device=/dev/video0 ! video/x-raw, width=1920, height=1080 ! nvvidconv ! video/x-raw(memory:NVMM) ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink")
    cap = cv2.VideoCapture("v4l2src device=/dev/video0 io-mode=2 ! image/jpeg, width=1920, height=1080 ! nvjpegdec ! video/x-raw ! videoconvert ! video/x-raw, format=BGR ! appsink")
    # cap = cv2.VideoCapture('udpsrc port=1234 ! "application/x-rtp, encoding-name=JPEG, payload=26" ! rtpjpegdepay ! jpegdec! appsink')
    if cap.isOpened():
        cv2.namedWindow("demo", cv2.WINDOW_GUI_NORMAL)
        while True:
            ret_val, img = cap.read();
            cv2.imshow('demo',img)
            cv2.waitKey(10)
    else:
     print( "camera open failed")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    read_cam()