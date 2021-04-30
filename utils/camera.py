import cv2
import threading


def jetson_gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
    
class WebcamVideoStream:
    """
    https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
    """

    def __init__(self, preprocess=None, pre_args=None):
        self.vid = None
        self.running = False
        self.preprocess = preprocess
        self.pre_args = pre_args
        
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

    def start(self, src, width=None, height=None):
        # initialize the video camera stream and read the first frame
        self.vid = cv2.VideoCapture(src)
        if not self.vid.isOpened():
            # camera failed
            raise IOError(("Couldn't open video file or webcam."))
        if width is not None and height is not None:
            self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.ret, self.frame = self.vid.read()
        if not self.ret:
            self.vid.release()
            raise IOError(("Couldn't open video frame."))
        if self.preprocess is not None:
            self.frame_p = self.preprocess(self.frame, *self.pre_args)
            
        # initialize the variable used to indicate if the thread should
        # check camera vid shape
        self.real_width = int(self.vid.get(3))
        self.real_height = int(self.vid.get(4))
        print("Start video stream with shape: {},{}".format(self.real_width, self.real_height))
        self.running = True

        # start the thread to read frames from the video stream
        t = threading.Thread(target=self.update, args=())
        t.setDaemon(True)
        t.start()
        return self

    def update(self):
        try:
            # keep looping infinitely until the stream is closed
            while self.running:
                # otherwise, read the next frame from the stream
                self.ret, self.frame = self.vid.read()
                if self.preprocess is not None:
                    self.frame_p = self.preprocess(self.frame, *self.pre_args)
        except:
            import traceback
            traceback.print_exc()
            self.running = False
        finally:
            # if the thread indicator variable is set, stop the thread
            self.vid.release()
        return

    def read(self):
        # return the frame most recently read
        if self.preprocess is not None:
            return self.frame, self.frame_p
        else:
            return self.frame

    def stop(self):
        self.running = False
        if self.vid.isOpened():
            self.vid.release()
        return