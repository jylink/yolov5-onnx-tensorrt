import cv2
import numpy as np
import time
from utils.yolo import YoloTRT, preprocess
from utils.plots import *
from utils.camera import *


def demo_image():
    img = cv2.imread("samples/0000001_02999_d_0000005.jpg")
    yolo = YoloTRT('models/s-default-352x608.yaml')
    
    for _ in range(50):
        boxes, confs, classes = yolo.detect(img)
        print('#det:', len(boxes), ', time(total, pre, infer, post):', *[int(t*1000) for t in yolo.timer])
        
    draw_results(img, boxes, confs, classes, names=yolo.names)
    cv2.imwrite('result.jpg', img)
                
def demo_camera():
    yolo = YoloTRT('models/s-default-352x608.yaml')
    input_src = jetson_gstreamer_pipeline(display_width=yolo.input_w, display_height=yolo.input_w*9//16)  # my camera is 16:9
    video_reader = WebcamVideoStream()
    video_reader.start(input_src)

    try:
        while video_reader.running:
            frame = video_reader.read()
            out = yolo.detect(frame)
            
            draw_results(frame, *out, names=yolo.names)
            plot_text((0, 50), frame, text='#det:{}, time(total, pre, infer, post): {} {} {} {}'.format(
                len(out[0]), *[int(t*1000) for t in yolo.timer]), color=(255, 0, 0))
            
            cv2.imshow('', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): # press q to quit
                break
    finally:
        video_reader.stop()
    
def demo_camera_hide_preprocess_delay():
    yolo = YoloTRT('models/s-default-352x608.yaml')
    input_src = jetson_gstreamer_pipeline(display_width=yolo.input_w, display_height=yolo.input_w*9//16)  # my camera is 16:9
    video_reader = WebcamVideoStream(preprocess=preprocess, pre_args=(yolo.input_h, yolo.input_w, not yolo.incl_norm))
    video_reader.start(input_src)

    try:
        while video_reader.running:
            frame, frame_p = video_reader.read()
            out = yolo.detect((frame, frame_p), pre=False)
            
            draw_results(frame, *out, names=yolo.names)
            plot_text((0, 50), frame, text='#det:{}, time(total, pre, infer, post): {} {} {} {}'.format(
                len(out[0]), *[int(t*1000) for t in yolo.timer]), color=(255, 0, 0))
            
            cv2.imshow('', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): # press q to quit
                break
    finally:
        video_reader.stop()
        
if __name__ == '__main__':
    demo_image()
    # demo_camera()
    # demo_camera_hide_preprocess_delay()
    
    