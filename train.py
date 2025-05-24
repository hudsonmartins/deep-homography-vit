import glob
import argparse
import torch
from tqdm import tqdm
import logging
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms.functional import to_tensor

from model import HomographyRegressor
from dataloader import HomographyDataset
from utils import visualize_homography_estimation


default_train_conf = {
    "epochs": 10,
    "lr": 0.001,
    "tensorboard_dir": "runs",
    "max_train_iter": 3000,
    "max_val_iter": 500,
}
default_train_conf = OmegaConf.create(default_train_conf)


def homography_loss(pred, target):
    return nn.functional.mse_loss(pred, target, reduction='mean')

def train_epoch(model, dataloader, optimizer, device, max_iters=None):
    model.train()
    total_loss = 0.0

    if max_iters is None or len(dataloader) < max_iters:
        max_iters = len(dataloader)
    
    for i, batch in enumerate(tqdm(dataloader, total=max_iters, desc="Training", leave=False)):
        if i >= max_iters:
            break
        image_pair = batch["image_pair"].to(device)
        gt_homography = batch["homography"].to(device)  # (B, 8)
        optimizer.zero_grad()
        data = {
            'view0': {'image': image_pair[:, :3, :, :]},
            'view1': {'image': image_pair[:, 3:, :, :]}
        }
        pred_homography = model(data)
        loss = homography_loss(pred_homography, gt_homography)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, device, max_iters=None):
    model.eval()
    total_loss = 0.0
    sample = []
    if max_iters is None or len(dataloader) < max_iters:
        max_iters = len(dataloader)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, total=max_iters, desc="Validation", leave=False)):
            if i >= max_iters:
                break
            image_pair = batch["image_pair"].to(device)
            gt_homography = batch["homography"].to(device)
            data = {
                'view0': {'image': image_pair[:, :3, :, :]},
                'view1': {'image': image_pair[:, 3:, :, :]}
            }
            pred_homography = model(data)
            loss = homography_loss(pred_homography, gt_homography)
            total_loss += loss.item()
            sample.append({'patches': batch["image_pair"][0],
                           'base_corners': batch["base_corners"][0],
                           'homography': batch["homography"][0],
                           'base_image': batch["base_image"][0],
                           'pred_homography': pred_homography[0]})        
    return total_loss / len(dataloader), sample


def train_model(model, train_loader, val_loader, optimizer, device, writer, config):
    best_val_loss = float('inf')
    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, max_iters=config.max_train_iter)
        logging.info(f"Epoch [{epoch+1}/{config.epochs}], Train Loss: {train_loss:.4f}")
        writer.add_scalar('Loss/train', train_loss, epoch)

        val_loss, sample = validate_epoch(model, val_loader, device, max_iters=config.max_val_iter)
        logging.info(f"Epoch [{epoch+1}/{config.epochs}], Val Loss: {val_loss:.4f}")
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        # Visualize samples
        for i in range(len(sample)):
            sample_patches = sample[i]['patches'].cpu().numpy()
            sample_base_image = sample[i]['base_image']
            sample_base_corners = sample[i]['base_corners'].cpu().numpy()
            sample_gt_deltas = sample[i]['homography'].cpu().numpy()
            sample_pred_deltas = sample[i]['pred_homography'].cpu().numpy()
    
            scale = val_loader.dataset.target_size / val_loader.dataset.patch_size
            sample_gt_deltas = sample_gt_deltas / scale
            sample_pred_deltas = sample_pred_deltas / scale
            fig = visualize_homography_estimation(sample_base_image, sample_base_corners, sample_gt_deltas, 
                                                sample_pred_deltas, sample_patches)
            fig = to_tensor(fig) 
            writer.add_image(f"Homography/Overlay_{i}", fig, global_step=epoch)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            logging.info(f"Model saved at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")
    
    writer.flush()
    return best_val_loss


def custom_collate(batch):
    batch_dict = {}
    for key in batch[0]:
        if key == "base_image":
            batch_dict[key] = [sample[key] for sample in batch]  # keep as list
        else:
            batch_dict[key] = default_collate([sample[key] for sample in batch])
    return batch_dict


def get_dataloaders(data_config):
    train_dataset = HomographyDataset(
        glob.glob(data_config.train.images_dir + '/*.jpg'),
        patch_size=data_config.train.patch_size,
        rho=data_config.train.rho,
        target_size=data_config.train.target_size,
        train=True
    )

    val_dataset = HomographyDataset(
        glob.glob(data_config.val.images_dir + '/*.jpg'),
        patch_size=data_config.val.patch_size,
        rho=data_config.val.rho,
        target_size=data_config.val.target_size,
        train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=data_config.batch_size, shuffle=data_config.train.shuffle, num_workers=data_config.num_workers, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=data_config.batch_size, shuffle=data_config.val.shuffle, num_workers=data_config.num_workers, collate_fn=custom_collate)

    return train_loader, val_loader


def main():
    # Load configuration
    config = OmegaConf.load(args.conf)
    config = OmegaConf.merge(default_train_conf, config)

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set up device
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = HomographyRegressor(config).to(device)

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Set up data loaders
    train_loader, val_loader = get_dataloaders(config.data)
    
    if(config.vit.freeze):
        for param in model.vit.parameters():
            param.requires_grad = False
        for param in model.regressor.parameters():
            param.requires_grad = True
    
    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir=config.tensorboard_dir)

    train_model(model, train_loader, val_loader, optimizer, device, writer, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, required=True)
    parser.add_argument("--use_cuda", action="store_true")

    args = parser.parse_args()
    main()