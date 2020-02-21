# use openCV to extract frames from a video
import cv2

def frame_capture(path):
    video = cv2.VideoCapture(path)
    # frame counter
    count = 0
    # sucess flag
    success = 1
    while success:
        success, image = video.read()
        # save frame as image
        cv2.imwrite("data/frames/rtel%d.jpg" % count, image)
        count += 1


if __name__ == '__main__':

    path = "data/tela.mp4"

    frame_capture(path)