import cv2
import random

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        plot_text(x, img, color=color, text=label, line_thickness=tl)
        
def plot_text(x, img, color=None, text=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1 = (int(x[0]), int(x[1]))
    for line in text.split('\n'):
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(line, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, line, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        c1 = (c1[0], c2[1])
        
def draw_results(img, boxes, confs, classes, names=None, boxcolor=(0,255,0), fontcolor=(0,0,255)):
    for box, conf, cls in zip(boxes, confs, classes):
        # draw rectangle
        x1, y1, x2, y2 = box
        conf = conf[0]
        cls_name = names[cls] if names is not None else cls
        cv2.rectangle(img, (x1, y1), (x2, y2), boxcolor, thickness=1, lineType=cv2.LINE_AA)
        # draw text
        cv2.putText(img, '%s %.2f' % (cls_name, conf), org=(x1, int(y1-10)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=fontcolor)
