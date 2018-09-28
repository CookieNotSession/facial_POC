import cv2

video_capture = cv2.VideoCapture(0)  # http://@10.50.197.220:8081/video.mjpg

while True:
    ret, frame = video_capture.read()
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
