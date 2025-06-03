import glob
import cv2
import tqdm
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from model import HomographyRegressor
from utils import custom_collate, visualize_homography_estimation, homography_to_four_points
from dataloader import HomographyDataset


def get_dataloader(images_dir):
    test_dataset = HomographyDataset(
        glob.glob(images_dir + '/*.jpg') + glob.glob(images_dir + '/*.png'),
        patch_size=512,
        rho=8,
        target_size=640,
        norm_factor=160,
        train=False
    )
    return DataLoader(test_dataset, batch_size=1, 
                            shuffle=False, num_workers=1, 
                            collate_fn=custom_collate)
    

def estimate_homography_orb(img1, img2):
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    if descriptors1 is None or descriptors2 is None:
        return None  # Not enough features detected

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 4:
        return None  # Not enough matches to estimate homography

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    return H


def estimate_homography_deep(model, dataloader, device='cpu'):
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            image_pair = batch["image_pair"].to(device)
            data = {
                'view0': {'image': image_pair[:, :3, :, :]},
                'view1': {'image': image_pair[:, 3:, :, :]}
            }
            pred_homography = model(data)
            scale = dataloader.dataset.target_size / dataloader.dataset.patch_size
            # visualize_homography_estimation
            base_image = batch["base_image"][0]
            base_corners = batch["base_corners"][0].numpy()
            gt_deltas = batch["homography"][0].numpy()*160/scale
            pred_deltas = pred_homography[0].cpu().numpy()*160/scale
            patches = image_pair[0].cpu().numpy()
            mce_deep =  np.mean(np.abs(gt_deltas.reshape(4, 2) - pred_deltas.reshape(4, 2)))

            fig = visualize_homography_estimation(base_image, 
                                                  base_corners, 
                                                  gt_deltas, 
                                                  pred_deltas, 
                                                  patches,
                                                  text=f"Deep MCE: {mce_deep:.2f}")
            with Image.fromarray(fig) as img:
                img.save(f"output/{i}_deep_homography_estimation.png")
            orb_homography = estimate_homography_orb(
                patches[:3, :, :].transpose(1, 2, 0),
                patches[3:, :, :].transpose(1, 2, 0)
            )
            if orb_homography is not None:
                orb_deltas = homography_to_four_points(orb_homography, base_corners)
                orb_deltas = orb_deltas.astype(np.float32) / scale
                mce_orb = np.mean(np.abs(gt_deltas.reshape(4, 2) - orb_deltas.reshape(4, 2)))
                fig = visualize_homography_estimation(base_image,
                                                    base_corners,
                                                    gt_deltas,
                                                    orb_deltas,
                                                    patches,
                                                    text=f"ORB MCE: {mce_orb:.2f}")
            else:
                mce_orb = 'Not enough features detected'
                fig = visualize_homography_estimation(base_image,
                                                    base_corners,
                                                    gt_deltas,
                                                    gt_deltas,
                                                    patches)
            with Image.fromarray(fig) as img:
                img.save(f"output/{i}_orb_homography_estimation.png")

def load_model(model_path):
    config = {
        'vit': {
            'image_size': [640,640],
            'pretrained': False,
            'patch_size': 16,
            'dim_emb': 384,
            'depth': 12,
            'heads': 6
        }
    }
    model = HomographyRegressor(OmegaConf.create(config))
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


if __name__ == "__main__":
    images_path = "data/queenscamp/sequences/16/breakage"
    model_path = "models/deep_homography_vit.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataloader = get_dataloader(images_path)
    model = load_model(model_path)
    estimate_homography_deep(model, dataloader, device)
