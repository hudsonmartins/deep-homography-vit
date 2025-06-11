import os
import torch
import glob
import argparse
import numpy as np
import random
import logging
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf

from iterators import get_iterator
from model import VORegressor
from utils import visualize_camera_poses

# Default training configuration
default_conf = {
    "epochs": 10,
    "lr": 0.001,
    "tensorboard_dir": "runs",
}
default_conf = OmegaConf.create(default_conf)

def compute_loss(pred, gt, criterion):
    loss = criterion(pred, gt.float())
    return loss


def pose_loss_norm(pred, gt, eps=1e-6):
    gt_R = gt[:, :3]
    gt_T = gt[:, 3:]
    pred_R = pred[:, :3]
    pred_T = pred[:, 3:]

    # Compute norms
    norm_pred_T = torch.norm(pred_T, dim=1, keepdim=True).clamp(min=eps)
    norm_gt_T = torch.norm(gt_T, dim=1, keepdim=True).clamp(min=eps)

    # Normalize translations
    pred_T_unit = pred_T / norm_pred_T
    gt_T_unit = gt_T / norm_gt_T

    # Compute translation and rotation losses
    trans_loss = F.l1_loss(pred_T_unit, gt_T_unit, reduction='mean')
    rot_loss = F.l1_loss(pred_R, gt_R, reduction='mean')

    return trans_loss + rot_loss


def val_epoch(model, val_loader, criterion, device, max_iters=None):
    model.eval()
    epoch_loss = 0.0
    sample = {}

    if max_iters is None or len(val_loader) < max_iters:
        max_iters = len(val_loader)

    with torch.no_grad():
        with tqdm(val_loader, unit="batch", total=max_iters, desc="Validating") as tepoch:
            for i, (images, gt, Ks) in enumerate(tepoch):
                if i >= max_iters:
                    break

                data = {
                    'view0': {'image': images[:, 0].to(device)},
                    'view1': {'image': images[:, 1].to(device)},
                    'K': Ks.to(device)
                }
                gt = gt.to(device)
                output = model(data)
                loss = compute_loss(output, gt, criterion)
                epoch_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

                if i == 0:
                    sample = {
                        'view0': data['view0']['image'].cpu(),
                        'view1': data['view1']['image'].cpu(),
                        'gt': gt.cpu(),
                        'pred': output.cpu()
                    }

    return epoch_loss / max_iters, sample


def train_epoch(model, train_loader, optimizer, device, criterion, max_iters=None):
    model.train()
    total_loss = 0.0
    
    if max_iters is None or len(train_loader) < max_iters:
        max_iters = len(train_loader)
    
    with tqdm(train_loader, unit="batch", total=max_iters, desc="Training") as tepoch:
        for i, (images, gt, Ks) in enumerate(tepoch):
            if i >= max_iters:
                break

            data = {
                'view0': {'image': images[:, 0].to(device)},
                'view1': {'image': images[:, 1].to(device)},
                'K': Ks.to(device)
            }
            
            gt = gt.to(device)
            output = model(data)
            loss = compute_loss(output, gt, criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())

    return total_loss / max_iters


def train_model(model, train_loader, val_loader, optimizer, device, writer, config):
    criterion = pose_loss_norm
    writer = SummaryWriter(log_dir=config.tensorboard_dir)
    best_loss = float("inf")
    for epoch in range(config.epochs):
        # Train the model for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion, max_iters=config.max_train_iter)
        logging.info(f"Epoch [{epoch+1}/{config.epochs}], Train Loss: {train_loss:.4f}")
        writer.add_scalar("train/loss", train_loss, epoch)

        # Validate the model
        val_loss, sample = val_epoch(model, val_loader, criterion, device, max_iters=config.max_val_iter)
        logging.info(f"Epoch [{epoch+1}/{config.epochs}], Validation Loss: {val_loss:.4f}")
        writer.add_scalar("val/loss", val_loss, epoch)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, "best_model.pth")
            logging.info(f"Saved best model at epoch {epoch} with loss {best_loss:.4f}")

            # draw camera poses and save to TensorBoard
            origin = torch.tensor([0, 0, 0, 0, 0, 0])
            fig_cameras = visualize_camera_poses(
                [origin[3:], sample['pred'][0, 3:].detach().cpu(), sample['gt'][0, 3:].detach().cpu()],
                [origin[:3], sample['pred'][0, :3].detach().cpu(), sample['gt'][0, :3].detach().cpu()],
                ["origin", "predicted", "ground truth"]
            )
            writer.add_figure("val/poses", fig_cameras, epoch)
    
    writer.flush()
    writer.close()
    return best_loss


def main():
    config = OmegaConf.load(args.conf)
    config = OmegaConf.merge(default_conf, config)
    device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Using device: {device}")

    # Initialize model
    model = VORegressor(config).to(device)
    if(config.vit.freeze):
        for param in model.encoder.parameters():
            param.requires_grad = False

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Set up data loaders
    train_loader = get_iterator(**config.data, train=True)
    val_loader = get_iterator(**config.data, train=False)   
    
    if(config.load_checkpoint):
        checkpoint = torch.load(config.load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info(f"Loaded checkpoint from {config.load_checkpoint}")


    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir=config.tensorboard_dir)

    train_model(model, train_loader, val_loader, optimizer, device, writer, config)

   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, required=True)
    parser.add_argument("--use_cuda", action="store_true")

    args = parser.parse_args()
    main()