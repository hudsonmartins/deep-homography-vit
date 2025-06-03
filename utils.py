import cv2
import matplotlib
import numpy as np
from io import BytesIO
from PIL import Image
from torch.utils.data._utils.collate import default_collate
from matplotlib import pyplot as plt
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