import torch
from d2l import torch as d2l

torch.set_printoptions(2) # 精简输出精度

#生成锚框函数
def multibox_prior(data, sizes, ratios):
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    offset_h, offset_w = 0.5, 0.5
    #归一化坐标统一用0 1表示可以规避具体像素尺寸问题
    #以后遇到不同尺寸输入却要统一处理的也可以有这个方法
    steps_h = 1.0 / in_height
    steps_w = 1.0 / in_width

    #生成锚框中心点
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w

    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
        sizes[0] * torch.sqrt(ratio_tensor[1:]))) * in_height / in_width
    
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
        sizes[0] / torch.sqrt(ratio_tensor[1:])))

    #获得半宽和半高然后作为偏移量
    #w和h包含了n+m-1
    #把每种锚框的宽高转换成"从中心点到四边的偏移量"，并复制到每个像素点
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2

    #包含了像素点个数
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                            dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations

    #升维匹配batch_size维度
    return output.unsqueeze(0)



img = d2l.plt.imread('/home/lanmei/prtice/char13/hand.jpg')
h, w = img.shape[:2]
# print(h, w)

X = torch.rand((1, 3, h, w))
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
# print(Y.shape)

def show_bboxes(axes, bboxes, labels=None, colors=None):
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj
    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])

    for i , bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], 
                      labels[i],va='center', ha='center', 
                      fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))

boxes = Y.reshape(h, w, 5, 4)
boxes[250, 250, 0, :]

d2l.set_figsize()
bbox_scale = torch.tensor((w, h, w, h))
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
's=0.75, r=0.5'])

# d2l.plt.show()

def box_iou(boxes1, boxes2):
    #计算宽，高
    box_area = lambda boxes:((boxes[:, 2]- boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))

    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    #广播机制的重要应用，实现不同位置的对应比较，逐元素比较的
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)

    inter_areas = inters[:, :, 0] * inters[:, :, 1]

    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas



def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    jaccard = box_iou(anchors, ground_truth)

    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
    #indices = tensor([1, 1, 0, 1, 0]) （每个锚框最爱的真实框编号）
    #max_ious = tensor([0.8, 0.2, 0.9, 0.4, 0.1]) （每个锚框的最高分）

    max_ious, indices = torch.max(jaccard, dim=1)

    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)

    box_j = indices[max_ious>=iou_threshold]
    #将及格锚框最匹配的【真实框的索引（编号）】赋值给【对应锚框的位置】
    anchors_bbox_map[anc_i] = box_j

    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)
        #列号
        box_idx = (max_idx % num_gt_boxes).long()
        #行号
        anc_idx = (max_idx // num_gt_boxes).long()

        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard

    return anchors_bbox_map


#偏移和缩放比例调整
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    c_anc=  d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset

#labels形状 (Batch_size, Num_objects, 5) 5：代表 [类别, x, y, w, h] （类别标签 + 4个坐标）。
def multibox_target(anchors, labels):
    #读取形状
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    #设置空列表备用
    batch_offset, batch_mask, batch_class_labels = [], [], []
    #获取锚框数
    device, num_anchors = anchors.device, anchors.shape[0]

    for i in range(batch_size):
        label = labels[i, :, :]

        #获得这个目标的匹配锚框
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device
        )

        #生成掩码用以屏蔽背景的坐标
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)

        #注意形状
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)

        #挑选背景和物体
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        #0号是背景，所以要加一
        #分配标签
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        #分配坐标
        assigned_bb[indices_true] = label[bb_idx, 1:]

        #偏移量转换，然后逐元素相乘
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask

        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    #把mask床图stack是为了屏蔽负样本的loss,只计算正样本的loss
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
        

def offset_inverse(anchors, offset_preds):
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:,:2] * anc[:, 2:] / 10) + anc[:,:2]
    pred_bbox_wh = torch.exp(offset_preds[:,2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh) , axis = 1)
    return pred_bbox


def nms(boxes, scores, iou_threshold):
    #返回的是索引
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = [] # 保留预测的边框

    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel == 1: break
        iou = box_iou(
            boxes[i, :].reshape(-1, 4),
            boxes[B[1:], :].reshape(-1, 4)
        ).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        #inds是B{1：}的数据，所以要加一
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,pos_threshold=0.009999999):
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range (batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        #沿着行方向去寻找最大置信度的类
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf ,nms_threshold)
        all_idx = torch.arange(num_anchors, dtype=torch.long,device=device)

        #关键在于后面的代码会用 combined 来对结果做填充。
        # SSD 的输出需要固定数量的检测框（通常是 num_anchors 个），
        # 不够的用背景框补齐。把 keep（真正检测到的目标）放在前面，
        # all_idx（所有锚框）放在后面，后续再取前 num_anchors 个

        #设置背景为-1
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts==1] 
        all_idx_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_idx_sorted]
        #设置阈值低的为-1
        conf, predicted_bb = conf[all_idx_sorted], predicted_bb[all_idx_sorted]
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]

        pred_info = torch.cat(
            class_id.unsqueeze(1),
            conf.unsqueeze(1),
            predicted_bb,
            dim=1
        )

        out.append(pred_info)
    return torch.stack(out)







        



