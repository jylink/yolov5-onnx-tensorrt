import cv2
import numpy as np
import yaml
import time
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda


def sigmoid(x):
    return np.reciprocal(np.exp(-x) + 1.0)
    
def xywh2xyxy(x, input_h, input_w, origin_h, origin_w):
    """
    description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    param:
        origin_h:   height of original image
        origin_w:   width of original image
        x:          A boxes tensor, each row is a box [center_x, center_y, w, h]
    return:
        y:          A boxes tensor, each row is a box [x1, y1, x2, y2]
    """
    y = np.zeros_like(x)
    r_w = input_w / origin_w
    r_h = input_h / origin_h
    if r_h > r_w:
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2 - (input_h - r_w * origin_h) / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2 - (input_h - r_w * origin_h) / 2
        y /= r_w
    else:
        y[:, 0] = x[:, 0] - x[:, 2] / 2 - (input_w - r_h * origin_w) / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2 - (input_w - r_h * origin_w) / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        y /= r_h

    return y

def make_grid(nx, ny):
    yv, xv = np.meshgrid(np.arange(ny), np.arange(nx))
    return np.stack((xv.T, yv.T), 2).reshape((1, 1, ny, nx, 2))

def nms_cpu(boxes, confs, classes, iou_thres):
    # cpu nms
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) 
    order = confs.flatten().argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where( ovr <= iou_thres)[0]
        order = order[inds + 1]
    boxes = boxes[keep]
    confs = confs[keep]
    classes = classes[keep]
    return boxes, confs, classes
    
def nms(boxes, confs, classes, iou_thres):
    # cuda nms, see faster-rcnn: https://github.com/rbgirshick/py-faster-rcnn/tree/master/lib/nms
    from utils.nms.gpu_nms import gpu_nms
    dets = np.concatenate((boxes, confs), 1)
    if len(dets) > 0:
        keep = gpu_nms(dets, iou_thres)
    else:
        keep = []
    boxes = boxes[keep]
    confs = confs[keep]
    classes = classes[keep]
    return boxes, confs, classes
    
def preprocess(image_raw, input_h, input_w, normalize):
    h, w, c = image_raw.shape
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    # Calculate widht and height and paddings
    r_w = input_w / w
    r_h = input_h / h
    if r_h > r_w:
        tw = input_w
        th = int(r_w * h)
        tx1 = tx2 = 0
        ty1 = int((input_h - th) / 2)
        ty2 = input_h - th - ty1
    else:
        tw = int(r_h * w)
        th = input_h
        tx1 = int((input_w - tw) / 2)
        tx2 = input_w - tw - tx1
        ty1 = ty2 = 0
    # Resize the image with long side while maintaining ratio
    image = cv2.resize(image, (tw, th))
    # Pad the short side with (128,128,128)
    image = cv2.copyMakeBorder(
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
    )
    image = image.astype(np.float32)
    # Normalize to [0,1]
    if normalize:
        image /= 255.0
    # HWC to CHW format:
    image = np.transpose(image, [2, 0, 1])
    # CHW to NCHW format
    image = np.expand_dims(image, axis=0)
    # Convert the image to row-major order, also known as "C order":
    image = np.ascontiguousarray(image)
    return image
    
def postprocess_cpu(outputs_raw, output_shapes, bs, nc, na, strides, anchor_grid, grids, input_h, input_w, origin_h, origin_w, conf_thres, iou_thres):
    """
    do the detect layer work here
    """
    outputs = []
    for o, shape in zip(outputs_raw[:3], output_shapes):
        outputs.append(o.reshape(shape))
            
    scaled = []
    for out in outputs:
        out = sigmoid(out)
        scaled.append(out)
    z = []
    for out, grid, stride, anchor in zip(scaled, grids, strides, anchor_grid):
        _, _, height, width, _ = out.shape
        out[..., 0:2] = (out[..., 0:2] * 2. - 0.5 + grid) * stride
        out[..., 2:4] = (out[..., 2:4] * 2) ** 2 * anchor
        out = out.reshape((bs, na * height * width, nc + 5))
        z.append(out)
    pred = np.concatenate(z, 1)
    xc = pred[..., 4] > conf_thres
    pred = pred[xc]
    
    boxes = xywh2xyxy(pred[..., 0:4], input_h, input_w, origin_h, origin_w)
    # best class only
    confs = np.amax(pred[:, 5:], 1, keepdims=True)
    classes = np.argmax(pred[:, 5:], axis=-1)
    return nms(boxes, confs, classes, iou_thres)
    
def postprocess(outputs_raw, bs, nc, input_h, input_w, origin_h, origin_w, conf_thres, iou_thres):
    """
    the detect layer work is done by tensorrt engine
    """
    pred = outputs_raw[-1].reshape(bs, -1, nc + 5)
    xc = pred[..., 4] > conf_thres
    pred = pred[xc]
    
    boxes = xywh2xyxy(pred[..., 0:4], input_h, input_w, origin_h, origin_w)
    # best class only
    confs = np.amax(pred[:, 5:], 1, keepdims=True)
    classes = np.argmax(pred[:, 5:], axis=-1)
    return nms(boxes, confs, classes, iou_thres)


class YoloTRT():
    def __init__(self, cfg):
        with open(cfg) as f:
            self.yaml = yaml.safe_load(f)
        
        # load tensorrt engine
        self.load_engine(self.yaml['model'])
        
        # parse cfg
        self.incl_norm = self.yaml['incl_norm']
        self.input_w = self.yaml['input_w']
        self.input_h = self.yaml['input_h']
        self.conf_thres = self.yaml['conf_thres']
        self.iou_thres = self.yaml['iou_thres']
        self.names = self.yaml['names']
        
        self.bs = 1  # now support batchsize=1 only
        self.nc = len(self.names)
        
        # For postprocess_cpu
        # self.strides = self.yaml['strides']
        # self.anchors = self.yaml['anchors']
        # assert len(self.strides) == len(self.anchors)
        # self.nl = len(self.anchors)
        # self.na = len(self.anchors[0]) // 2
        # self.anchor_grid = np.array(self.anchors).reshape(self.nl, 1, self.na, 1, 1, 2)
        # self.output_shapes = [[self.bs, self.na, self.input_h // s, self.input_w // s, self.nc + 5] for s in self.strides]
        # self.grids = [make_grid(self.input_w // s, self.input_h // s) for s in self.strides]
        
        self.timer = [0] * 4  # total, pre, infer, post 
        
        # warming up
        for _ in range(10):
            dummy_in = np.random.randint(0, 255, size=(self.input_h, self.input_w, 3), dtype=np.uint8)
            self.detect(dummy_in)
            
    def load_engine(self, model):
        # load tensorrt engine
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        with open(model, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()
        # allocate memory
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                inputs.append({ 'host': host_mem, 'device': device_mem })
            else:
                outputs.append({ 'host': host_mem, 'device': device_mem })
        # save to class
        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings
        self.stream = stream
        
    def detect(self, img, pre=True, post=True):
        t1 = time.time()
        
        if pre: 
            origin_h, origin_w, _ = img.shape
            img = preprocess(img, input_h=self.input_h, input_w=self.input_w, normalize=not self.incl_norm)
        else:
            img_ori, img = img
            origin_h, origin_w, _ = img_ori.shape
            
        t2 = time.time()
        
        out = self.inference(img)
        
        t3 = time.time()
        
        if post:
            # out = postprocess_cpu(out, output_shapes=self.output_shapes, bs=self.bs, nc=self.nc, na=self.na, strides=self.strides, 
                # anchor_grid=self.anchor_grid, grids=self.grids, input_h=self.input_h, input_w=self.input_w, 
                # origin_h=origin_h, origin_w=origin_w, conf_thres=self.conf_thres, iou_thres=self.iou_thres)
            out = postprocess(out, bs=self.bs, nc=self.nc, input_h=self.input_h, input_w=self.input_w, 
                origin_h=origin_h, origin_w=origin_w, conf_thres=self.conf_thres, iou_thres=self.iou_thres)
                
        t4 = time.time()
        
        self.timer[1] = self.timer[1] * 0.9 + (t2 - t1) * 0.1
        self.timer[2] = self.timer[2] * 0.9 + (t3 - t2) * 0.1
        self.timer[3] = self.timer[3] * 0.9 + (t4 - t3) * 0.1
        self.timer[0] = sum(self.timer[1:])
        
        return out

    def inference(self, img):
        # copy img to input memory
        self.inputs[0]['host'] = np.ravel(img)
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.handle)
       # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()
        return [out['host'] for out in self.outputs]
        
        
