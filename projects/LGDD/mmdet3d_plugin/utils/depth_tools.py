import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torchvision.utils import save_image
import cv2
import numpy as np

def clip_sigmoid(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    y = torch.sigmoid(x)
    return torch.clamp(y, min=eps, max=1 - eps)
def draw_true_depth(depth_labels, min_max):
    B = depth_labels.shape[0]
    batch_depth = []
    for i in range(B): 
        draw_gt_depth = depth_labels[i].cpu().detach().numpy()
        draw_gt_depth = (draw_gt_depth - min_max[0][i])/(min_max[1][i] - min_max[0][i] + 1e-3)
        draw_gt_depth = np.clip(draw_gt_depth, 0.01, 0.99)
        draw_gt_depth = (draw_gt_depth*255).astype(np.uint8).transpose(1, 2, 0)
        draw_gt_depth = cv2.applyColorMap(cv2.convertScaleAbs(draw_gt_depth, alpha=5), cv2.COLORMAP_JET)
        batch_depth.append(draw_gt_depth)
    batch_depth = np.concatenate(batch_depth, axis=1)
    return batch_depth

def draw_prob_depth(depth_prob, resize, cam_depth_range, min_max):
    raw_depth = torch.arange(cam_depth_range[0], cam_depth_range[1], cam_depth_range[2]).to(depth_prob.device)
    B, H, W, bins = depth_prob.shape
    index = depth_prob.max(dim=-1)[1] # B H W
    vis_depth = torch.gather(raw_depth.view(1, 1, -1).expand(B, H, W, -1), -1, index.unsqueeze(1))
    vis_depth = F.interpolate(vis_depth, size=(resize[0], resize[1]), mode='bilinear', align_corners=False)
    min = torch.tensor(min_max[0]).view(-1, 1, 1, 1).repeat(1, 1, resize[0], resize[1]).to(depth_prob.device)
    max = torch.tensor(min_max[1]).view(-1, 1, 1, 1).repeat(1, 1, resize[0], resize[1]).to(depth_prob.device)
    vis_depth = torch.clip((vis_depth - min)/(max - min), 0.01, 0.99)
    vis_depth = (vis_depth.cpu().detach().numpy()*255).astype(np.uint8).transpose(0, 2, 3, 1)
    batch_depth = []
    for i in range(B):
        vis_depth_single = vis_depth[i]
        vis_depth_single = cv2.applyColorMap(cv2.convertScaleAbs(vis_depth_single, alpha=5), cv2.COLORMAP_JET)
        batch_depth.append(vis_depth_single)
    batch_depth = np.concatenate(batch_depth, axis=1)
    # cv2.imwrite('batch_depth.png', batch_depth)
    return batch_depth

def draw_sum_depth(depth_prob, resize, cam_depth_range, min_max):
    show = B = depth_prob.shape[0]
    raw_depth = torch.arange(cam_depth_range[0], cam_depth_range[1], cam_depth_range[2]).to(depth_prob.device)
    vis_depth = torch.sum(raw_depth.view(1,1,1,-1)*depth_prob[:show], dim=-1).unsqueeze(1)
    vis_depth = F.interpolate(vis_depth, size=(resize[0], resize[1]), mode='bilinear', align_corners=False)
    min = torch.tensor(min_max[0]).view(-1, 1, 1, 1).repeat(1, 1, resize[0], resize[1]).to(depth_prob.device)
    max = torch.tensor(min_max[1]).view(-1, 1, 1, 1).repeat(1, 1, resize[0], resize[1]).to(depth_prob.device)
    vis_depth = torch.clip((vis_depth - min)/(max - min), 0.01, 0.99)
    vis_depth = (vis_depth.cpu().detach().numpy()*255).astype(np.uint8).transpose(0, 2, 3, 1)
    batch_depth = []
    for i in range(B):
        vis_depth_single = vis_depth[i]
        vis_depth_single = cv2.applyColorMap(cv2.convertScaleAbs(vis_depth_single, alpha=5), cv2.COLORMAP_JET)
        batch_depth.append(vis_depth_single)
    batch_depth = np.concatenate(batch_depth, axis=1)
    return batch_depth

def get_downsample_depths_torch(depth, down, processing='min'):
    B, C, H, W = depth.shape
    depth = depth.view(B, H//down, down, W//down, down, 1)
    depth = depth.permute(0, 1, 3, 5, 2, 4).contiguous()
    depth = depth.view(-1, down * down)
    depth_tmp = torch.where(depth == 0.0, 1e5 * torch.ones_like(depth), depth)
    if processing == 'min': 
        depth = torch.min(depth_tmp, dim=-1).values
    if processing == 'max': 
        depth = torch.max(depth_tmp, dim=-1).values
    if processing == 'mean': 
        depth = torch.mean(depth_tmp, dim=-1)
    depth = depth.view(B, C, H//down, W//down)
    return depth
    
def generate_guassian_depth_target(depth, stride, cam_depth_range, constant_std=None):
    depth = depth.flatten(0, 1)
    B, tH, tW = depth.shape
    kernel_size = stride
    center_idx = kernel_size * kernel_size // 2
    H = tH // stride
    W = tW // stride
    
    unfold_depth = F.unfold(depth.unsqueeze(1), kernel_size, dilation=1, padding=0, stride=stride) # B, Cxkxk, HxW, here C=1
    unfold_depth = unfold_depth.view(B, -1, H, W).permute(0, 2, 3, 1).contiguous() # BN, H, W, kxk
    valid_mask = (unfold_depth != 0) # BN, H, W, kxk
    
    if constant_std is None:
        valid_mask_f = valid_mask.float() # BN, H, W, kxk
        valid_num = torch.sum(valid_mask_f, dim=-1) # BN, H, W
        valid_num[valid_num == 0] = 1e10
        
        mean = torch.sum(unfold_depth, dim=-1) / valid_num
        var_sum = torch.sum(((unfold_depth - mean.unsqueeze(-1))**2) * valid_mask_f, dim=-1) # BN, H, W
        std_var = torch.sqrt(var_sum / valid_num)
        std_var[valid_num == 1] = 1 # set std_var to 1 when only one point in patch
    else:
        std_var = torch.ones((B, H, W)).type_as(depth).float() * constant_std

    unfold_depth[~valid_mask] = 1e10
    min_depth = torch.min(unfold_depth, dim=-1)[0] # BN, H, W, min_depth in stridexstride block
    loss_valid_mask = ~(min_depth == 1e10)
    min_depth[min_depth == 1e10] = 0
    
    # x in raw depth 
    x = torch.arange(cam_depth_range[0] - cam_depth_range[2] / 2, cam_depth_range[1], cam_depth_range[2])
    # normalized by intervals
    dist = Normal(min_depth / cam_depth_range[2], std_var / cam_depth_range[2]) # BN, H, W, D
    # dist = Normal(min_depth, std_var / cam_depth_range[2]) # BN, H, W, D
    cdfs = []
    for i in x:
        cdf = dist.cdf(i)
        cdfs.append(cdf)
    
    cdfs = torch.stack(cdfs, dim=-1)
    depth_dist = cdfs[..., 1:] - cdfs[...,:-1]

    return depth_dist, min_depth

def sobel_operator(img):
    """
    Apply Sobel operator to compute image gradients.
    
    Args:
        img: torch.Tensor, the input image with shape (B, C, H, W)
    
    Returns:
        grad: torch.Tensor, the computed gradient magnitude with shape (B, 1, H, W)
    """
    sobel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).to(img.device).view(1, 1, 3, 3)
    sobel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).to(img.device).view(1, 1, 3, 3)

    grad_x = F.conv2d(img, sobel_x, padding=1, groups=img.shape[1])
    grad_y = F.conv2d(img, sobel_y, padding=1, groups=img.shape[1])
    grad = torch.sqrt(grad_x**2 + grad_y**2)
    
    return grad
def edge_aware_smoothness_loss(img, depth):
    """
    Edge-aware smoothness loss.
    
    Args:
        img: torch.Tensor, the input image with shape (B, C, H, W)
        depth: torch.Tensor, the estimated depth map with shape (B, 1, H, W)
    
    Returns:
        torch.Tensor, the computed edge-aware smoothness loss.
    """
    b, c, h, w = depth.shape
    img_down = F.interpolate(img, (h, w), mode='bilinear', align_corners=True)
    img_down = torch.mean(img_down, dim=1, keepdim=True)
    
    # Compute gradients for image and depth
    grad_img = sobel_operator(img_down)
    grad_depth = sobel_operator(depth)
    
    # Weight depth gradients by image gradients
    weight = torch.exp(-grad_img)
    
    # Compute the edge-aware smoothness loss
    smoothness_loss = torch.mean(weight * grad_depth)
    
    return smoothness_loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    gauss = torch.exp(gauss) 
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        return 1 - ssim(img1, img2, self.window_size, self.size_average)