import cv2
import torch
import matplotlib
import numpy as np
from io import BytesIO
from PIL import Image
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from torch.utils.data._utils.collate import default_collate
matplotlib.use('Agg')  


def custom_collate(batch):
    batch_dict = {}
    for key in batch[0]:
        if key == "base_image":
            batch_dict[key] = [sample[key] for sample in batch]  # only visualization, keep as list
        else:
            batch_dict[key] = default_collate([sample[key] for sample in batch])
    return batch_dict


def four_point_to_homography(corner_deltas, base_corners):
    base_corners = np.array(base_corners, dtype=np.float32).reshape(4, 2)
    corner_deltas = np.array(corner_deltas, dtype=np.float32).reshape(4, 2)
    return cv2.getPerspectiveTransform(base_corners, base_corners + corner_deltas)


def homography_to_four_points(homography, base_corners):
    base_corners = np.array(base_corners, dtype=np.float32).reshape(4, 2)
    transformed_corners = cv2.perspectiveTransform(base_corners.reshape(-1, 1, 2), homography)
    return (transformed_corners - base_corners.reshape(-1, 1, 2)).reshape(4, 2).astype(np.int32)


def visualize_homography_estimation(base_image, base_corners, gt_deltas, pred_deltas, patches, text=None):
    
    base_corners = np.array(base_corners, dtype=np.int32)
    gt_deltas = np.array(gt_deltas, dtype=np.int32).reshape(4, 2)
    pred_deltas = np.array(pred_deltas, dtype=np.int32).reshape(4, 2)

    base_image_bgr = base_image[:, :, ::-1].astype(np.uint8)  # Convert RGB to BGR for OpenCV

    # warp the base image using the ground truth deltas
    gt_H = four_point_to_homography(gt_deltas, base_corners)
    warped_image_bgr = cv2.warpPerspective(base_image_bgr, gt_H, (base_image.shape[1], base_image.shape[0]))

    # Draw on base image
    for i in range(4):
       pt1 = tuple(base_corners[i])
       pt2 = tuple(base_corners[(i + 1) % 4])
       cv2.line(base_image_bgr, pt1, pt2, (0, 255, 0), 2)

    # Draw GT and predicted corners on warped image
    for i in range(4):
        pt1 = tuple(base_corners[i] + gt_deltas[i])
        pt2 = tuple(base_corners[(i + 1) % 4] + gt_deltas[(i + 1) % 4])
        cv2.line(warped_image_bgr, pt1, pt2, (0, 255, 0), 2)

        pt1_pred = tuple(base_corners[i] + pred_deltas[i])
        pt2_pred = tuple(base_corners[(i + 1) % 4] + pred_deltas[(i + 1) % 4])
        cv2.line(warped_image_bgr, pt1_pred, pt2_pred, (0, 0, 255), 2)

    base_image_rgb = base_image_bgr[:, :, ::-1]  # Convert BGR back to RGB for pkots
    warped_image_rgb = warped_image_bgr[:, :, ::-1]  

    patch_A = patches[:3, :, :].transpose(1, 2, 0)
    patch_B = patches[3:, :, :].transpose(1, 2, 0) 
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.imshow(base_image_rgb)
    plt.title('Base Image with Original Corners')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(warped_image_rgb)
    plt.title('Warped Image with Predicted Corners')
    if text:
        plt.text(0.5, -0.1, text, ha='center', va='top', transform=plt.gca().transAxes, fontsize=12, color='red')   
    plt.axis('off')

    # Draw patches
    plt.subplot(2, 2, 3)
    plt.imshow(patch_A)
    plt.title('Patch A')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(patch_B)
    plt.title('Patch B')
    plt.axis('off')

    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf).convert('RGB')
    image_np = np.array(image)
    plt.close()

    return image_np

def visualize_camera_poses(trans, rot, labels):
    fig, axs = plt.subplots(2, 1, figsize=(6, 6), dpi=700)
    for (t, r, label) in zip(trans, rot, labels):
        # Extracting translation and rotation components
        xy = t[:2]
        xz = t[[0, 2]]
        rotation = R.from_euler('ZYX', r).as_matrix()

        # Plot 1 shows XY plane
        axs[0].quiver(*xy, rotation[0, 0], rotation[1, 0], headaxislength=0, headwidth=0, headlength=0, color='r', label='X axis')
        axs[0].quiver(*xy, rotation[0, 1], rotation[1, 1], headaxislength=0, headwidth=0, headlength=0, color='g', label='Y axis')
        axs[0].text(xy[0], xy[1], label, fontsize=12, color='black')
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')

        # Plot 2 shows XZ plane
        axs[1].quiver(*xz, rotation[0, 0], rotation[2, 0], headaxislength=0, headwidth=0, headlength=0, color='r', label='X axis')
        axs[1].quiver(*xz, rotation[0, 2], rotation[2, 2], headaxislength=0, headwidth=0, headlength=0, color='b', label='Z axis')
        axs[1].text(xz[0], xz[1], label, fontsize=12, color='black')
        axs[1].set_xlabel('X')
        axs[1].set_ylabel('Z')

    for i in range(2):
        axs[i].grid(True)
        fig.tight_layout(pad=0.5)
    plt.close()
    return fig

def make_intrinsics_layer(height, width, Ks):
    """
    Create a batch of intrinsic parameter layers for different cameras
    """
    
    # Create base grid with 0.5 offset for pixel center
    x_coords = torch.arange(width, dtype=torch.float32, device=Ks.device) + 0.5
    y_coords = torch.arange(height, dtype=torch.float32, device=Ks.device) + 0.5
    xx, yy = torch.meshgrid(x_coords, y_coords, indexing='xy')  # (H, W)
    
    # Expand to batch dimensions (B, H, W)
    batch_size = Ks.size(0)
    xx = xx.unsqueeze(0).expand(batch_size, -1, -1)  # (B, H, W)
    yy = yy.unsqueeze(0).expand(batch_size, -1, -1)  # (B, H, W)
    
    # Expand intrinsics to match dimensions
    fx = Ks[:, 0].unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
    fy = Ks[:, 1].unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
    ox = Ks[:, 2].unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
    oy = Ks[:, 3].unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
        
    # Calculate normalized coordinates
    kcx = (xx - ox) / fx
    kcy = (yy - oy) / fy
    
    # Stack into final tensor (B, 2, H, W)
    return torch.stack([kcx, kcy], dim=1)