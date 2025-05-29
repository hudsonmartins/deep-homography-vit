import cv2
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils import visualize_homography_estimation


class HomographyDataset(Dataset):
    def __init__(self, image_paths, patch_size=128, rho=32, target_size=640, train=True):
        """
        Args:
            image_paths: list of image paths
            patch_size: size of original square patch before resizing
            rho: max perturbation in pixels
            target_size: final resized patch size
        """
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.rho = rho
        self.target_size = target_size
        self.train = train
        if(self.train):
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)], p=0.5),
                transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 1.0))], p=0.3),
                transforms.Resize((target_size, target_size)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((target_size, target_size)),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        with Image.open(img_path) as pil_img:
            pil_img = pil_img.convert('RGB')
            img = np.array(pil_img)

        h, w, _ = img.shape
        ps = self.patch_size
        margin = self.rho

        if w <= ps + 2 * margin or h <= ps + 2 * margin:
            pad_h = max(0, ps + 2 * margin - h + 1)
            pad_w = max(0, ps + 2 * margin - w + 1)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            h, w, _ = img.shape
        
        # Random top-left corner
        x = np.random.randint(margin, w - ps - margin)
        y = np.random.randint(margin, h - ps - margin)

        # Base corners (Patch A)
        base_corners = np.array([
            [x, y],
            [x + ps, y],
            [x + ps, y + ps],
            [x, y + ps]
        ], dtype=np.float32)

        # Perturb the corners
        deltas = np.random.uniform(-self.rho, self.rho, size=(4, 2)).astype(np.float32)
        perturbed_corners = base_corners + deltas

        # Compute warp matrix
        H_AB = cv2.getPerspectiveTransform(base_corners, perturbed_corners)
        H_BA = np.linalg.inv(H_AB)
        warped_img = cv2.warpPerspective(img, H_BA, (w, h))
        # Crop the patches
        patch_A = img[y:y+ps, x:x+ps,:]
        patch_B = warped_img[y:y+ps, x:x+ps,:]

        # Convert to PIL for torchvision transforms
        patch_A = Image.fromarray(patch_A)
        patch_B = Image.fromarray(patch_B)

        if self.train:
            worker_info = torch.utils.data.get_worker_info()
            seed =  worker_info.seed if worker_info else random.randint(0, 2**32 - 1)
            random.seed(seed)
            torch.manual_seed(seed)
            patch_A_tensor = self.transform(patch_A)
            random.seed(seed)
            torch.manual_seed(seed)
            patch_B_tensor = self.transform(patch_B)
        else:
            patch_A_tensor = self.transform(patch_A)
            patch_B_tensor = self.transform(patch_B)
        
        # Stack patches
        input_tensor = torch.cat([patch_A_tensor, patch_B_tensor], dim=0)  # (6, H, W)

        # Scale deltas to match resized image
        scale = self.target_size / self.patch_size
        deltas_scaled = deltas * scale
        label = deltas_scaled.reshape(-1)  # (8,)

        return {
            "image_pair": input_tensor,  # (6, target_size, target_size)
            "homography": torch.from_numpy(label).float(),
            "base_corners": base_corners,  # (4, 2)
            "base_image": img  # RGB image as numpy array for visualization
        }
