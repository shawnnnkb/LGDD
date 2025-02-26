import torch
from shapely.geometry import Polygon

def calculate_rotated_iou(box1, box2):
    """
    计算两个旋转框的IoU值
    box1: [N1, 5] (x, y, w, h, angle in radians)
    box2: [N2, 5] (x, y, w, h, angle in radians)
    """
    def get_rotated_corners(box):
        """
        根据中心点、宽高和角度获取旋转框的四个顶点
        box: [N, 5]
        返回: [N, 4, 2] (四个顶点的坐标)
        """
        cx, cy, w, h, angle = box.T
        # 角度的cos和sin
        cos, sin = torch.cos(angle), torch.sin(angle)
        # 矩形的相对坐标
        dx = torch.tensor([[-0.5, 0.5, 0.5, -0.5]], device=box.device) * w[:, None]
        dy = torch.tensor([[-0.5, -0.5, 0.5, 0.5]], device=box.device) * h[:, None]
        # 旋转后的位置
        x_rot = cos[:, None] * dx - sin[:, None] * dy
        y_rot = sin[:, None] * dx + cos[:, None] * dy
        # 加回中心点偏移
        corners = torch.stack([cx[:, None] + x_rot, cy[:, None] + y_rot], dim=-1)
        return corners  # [N, 4, 2]

    def calculate_polygon_iou(corners1, corners2):
        """
        计算两个多边形的IoU
        corners1: [4, 2]
        corners2: [4, 2]
        """
        poly1 = Polygon(corners1)
        poly2 = Polygon(corners2)
        if not poly1.is_valid or not poly2.is_valid:
            return 0.0
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.area + poly2.area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    # 获取旋转框的顶点
    corners1 = get_rotated_corners(box1)  # [N1, 4, 2]
    corners2 = get_rotated_corners(box2)  # [N2, 4, 2]

    # 逐一计算IoU
    iou_matrix = torch.zeros((len(box1), len(box2)), device=box1.device)
    for i in range(len(box1)):
        for j in range(len(box2)):
            iou_matrix[i, j] = calculate_polygon_iou(corners1[i].cpu().numpy(), corners2[j].cpu().numpy())

    return iou_matrix

def filter_boxes_by_iou(gt_bboxes_3d_semi, gt_bboxes_3d, gt_labels_3d, threshold=0.25):
    """
    根据IoU筛选框
    tensor1: [N1, 5] (x, y, w, h, angle)
    tensor2: [N2, 5] (x, y, w, h, angle)
    """
    gt_bboxes_3d_result = []
    gt_labels_3d_result = []
    for boxes_1, boxes_2, label_2 in zip(gt_bboxes_3d_semi, gt_bboxes_3d, gt_labels_3d):
        tensor_1 = torch.cat([boxes_1.tensor[:, 0:2], boxes_1.tensor[:, 3:5], boxes_1.tensor[:, 6:7]], dim=-1)
        tensor_2 = torch.cat([boxes_2.tensor[:, 0:2], boxes_2.tensor[:, 3:5], boxes_2.tensor[:, 6:7]], dim=-1)
        iou = calculate_rotated_iou(tensor_1, tensor_2)  # [N1, N2]
        max_iou, _ = torch.max(iou, dim=0)            # [N2]
        keep_mask = max_iou > threshold               # [N2]
        gt_bboxes_3d_result.append(boxes_2[keep_mask])
        gt_labels_3d_result.append(label_2[keep_mask])
    return gt_bboxes_3d_result, gt_labels_3d_result

if __name__ == "__main__":
    # 示例数据
    tensor1 = torch.tensor([[0, 0, 2, 2, 0], [1, 1, 2, 2, 0]], dtype=torch.float32)
    tensor2 = torch.tensor([[0, 0, 2, 2, 0], [1, 1, 2, 2, 0], [1.5, 1.5, 1, 1, 0]], dtype=torch.float32)

    # 筛选框
    result = filter_boxes_by_iou(tensor1, tensor2, threshold=0.25)
    print(result)
