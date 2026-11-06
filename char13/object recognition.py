import torch
import numpy as np
from d2l import torch as d2l
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
def box_corner_to_center(boxes):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx =(x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes



def box_center_to_corner(boxes):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]

boxes = torch.tensor((dog_bbox, cat_bbox))
print(box_center_to_corner(box_corner_to_center(boxes)) == boxes)

def bbox_to_rect(bbox, color):
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)


# img = d2l.plt.imread('char13/hand.jpg')
# img = np.array(Image.fromarray(img).resize((1000, 1000)))
# fig = d2l.plt.imshow(img)


# d2l.plt.show() 


img_pil = Image.open('char13/hand.jpg')

# 2. 👇 真正改尺寸的是这行！把图片缩放到 500x500
img_pil = img_pil.resize((1000, 1000)) 

# 3. 👇 这行只是把改好尺寸的 PIL 图片转成 NumPy 数组（画图必须的格式），它不改尺寸
img = np.array(img_pil)

# 4. 画图
fig = d2l.plt.imshow(img)
# 注意：这里你用原图的坐标画框，因为图片缩小了，框可能会错位
# 如果是原图 1000x1000 变成了 500x500，坐标得乘 0.5
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
d2l.plt.show()


