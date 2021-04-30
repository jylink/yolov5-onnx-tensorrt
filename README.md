# yolov5-onnx-tensorrt

Yolov5 in Pytorch (.pt) --> .onnx --> TensorRT engine (.trt)

Another way without using onnx is: yolov5 --> tensorrt api (see [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx)), but everytime you have to modify the tensorrt api after you change the model structure



# Training

Train your model based on [ultralytics/yolov5](https://github.com/ultralytics/yolov5)

Also supports:

* [custom model structure](https://github.com/jylink/yolov5-mobilenetv3)
* [knowledge distillation](https://github.com/jylink/yolov5-distillation)
* [model pruning](https://github.com/jylink/yolov5-pruning)



# Export onnx

yolov5 has an official [onnx export](https://github.com/ultralytics/yolov5/issues/251). Export your onnx with `--grid --simplify` to include the detect layer (otherwise you have to config the anchor and do the detect layer work during postprocess)

Q: I can't export onnx with `--grid --simplify` / the exported onnx is broken

A: change the following lines in ` yolov5/models/yolo.py `

```
# y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
# y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
# z.append(y.view(bs, -1, self.no))
xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(bs, self.na, 1, 1, 2)  # wh
rest = y[..., 4:]
y_ = torch.cat((xy, wh, rest), -1)
z.append(y_.view(bs, -1, self.no))
```



# Convert to tensorrt engine

```
python tools/export_trt.py --model xxx.onnx --out xxx.trt --fp 16
```

Q: I can't convert onnx to trt

A: same as the q&a of previous section



# Inference

Run demo:

```
python demo.py
```

Performance on jetson nano:

| model             | input     | preprocess | inference | postprocess | total |
| ----------------- | --------- | ---------- | --------- | ----------- | ----- |
| yolov5s (352x608) | 1080x1920 | 8ms        | 44ms      | 3ms         | 56ms  |



# Other tricks

* independent `preprocess` and `postprocess` which could hide into other thread
* add `x=x/255` to  `Model.forward_once` in ` yolov5/models/yolo.py `, set `incl_norm: true` in your `xxx.yaml`
* fast nms using cuda (setup: `cd utils && python setup_nms.py build`)



# Reference

* https://github.com/SeanAvery/yolov5-tensorrt
* https://github.com/bei91/yolov5-onnx-tensorrt
* https://github.com/wang-xinyu/tensorrtx
* https://github.com/rbgirshick/py-faster-rcnn/tree/master/lib/nms
* https://github.com/ultralytics/yolov5/issues/251
* https://github.com/ultralytics/yolov5/issues/2558
* https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/



