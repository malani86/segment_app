# train.py

import logging
import cv2
import os
import numpy as np
import torch
torch.cuda.empty_cache()
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from utils.data_loader import SegmentationDataset
from models.model_2 import UNetDC
from utils.metrics_DC import combined_loss, dice_coef, calculate_metrics, plot_binary_confusion_matrix_with_metrics, focal_dice_loss

from sklearn.model_selection import train_test_split
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from torchsummary import summary
except ImportError:  # pragma: no cover - optional dependency
    summary = None
    logging.getLogger(__name__).warning(
        "torchsummary is not installed. Model summary will be skipped.")





# Setup Logging
# -------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def create_difference_map(true_mask, pred_mask):
    """
    true_mask, pred_mask: 2D numpy arrays (binary) of the same shape, 
                          e.g. shape (H, W), values 0 or 1.
    Returns an (H, W, 3) RGB array showing:
      - Yellow where both are 1
      - Red where only true is 1
      - Green where only pred is 1
      - Black otherwise
    """

    # Ensure they are binary (0 or 1)
    true_mask_bin = (true_mask > 0).astype(np.uint8)
    pred_mask_bin = (pred_mask > 0).astype(np.uint8)

    # Intersection (common droplets)
    common = (true_mask_bin & pred_mask_bin)  # both are 1
    # Only predicted
    only_pred = (pred_mask_bin & (1 - true_mask_bin))
    # Only in true
    only_true = (true_mask_bin & (1 - pred_mask_bin))
    
    # Create an empty color image
    h, w = true_mask.shape
    diff_map = np.zeros((h, w, 3), dtype=np.uint8)

    # Yellow = (255, 255, 0)
    diff_map[common == 1] = (255, 255, 0)

    # Green = (0, 255, 0)
    diff_map[only_pred == 1] = (0, 255, 0)

    # Red = (255, 0, 0)
    diff_map[only_true == 1] = (255, 0, 0)
    
    return diff_map
def overlay_difference(original_img, diff_map):
    """
    Overlays the difference map on the original image.
    Non-black pixels in diff_map replace those in the original.
    """
    overlayed = original_img.copy()
    non_black = np.any(diff_map != [0, 0, 0], axis=-1)
    overlayed[non_black] = diff_map[non_black]
    return overlayed


##############################
# COUNT COLOR REGIONS
##############################
def count_color_regions_in_memory(diff_map_rgb):
    """
    Given an (H, W, 3) difference map (RGB) from create_difference_map,
    threshold each color to find distinct connected regions (yellow, red, green, black).
    
    Returns a dict with:
      {
        'yellow_blobs': int,  # true positives
        'red_blobs':    int,  # false negatives
        'green_blobs':  int,  # false positives
        'black_blobs':  int,  # true negatives
      }
    """
    R = diff_map_rgb[:,:,0]
    G = diff_map_rgb[:,:,1]
    B = diff_map_rgb[:,:,2]

    # For pure colors: (255,255,0) => R>200, G>200, B<50 => etc.
    yellow_mask = np.zeros_like(R, dtype=np.uint8)
    yellow_mask[(R>200) & (G>200) & (B<50)] = 255

    red_mask = np.zeros_like(R, dtype=np.uint8)
    red_mask[(R>200) & (G<50) & (B<50)] = 255

    green_mask = np.zeros_like(R, dtype=np.uint8)
    green_mask[(R<50) & (G>200) & (B<50)] = 255

    black_mask = np.zeros_like(R, dtype=np.uint8)
    black_mask[(R<50) & (G<50) & (B<50)] = 255

    def count_blobs(binary_mask):
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        return num_labels - 1  # minus 1 for background label

    return {
        "yellow_blobs": count_blobs(yellow_mask),  # TP
        "red_blobs":    count_blobs(red_mask),     # FN
        "green_blobs":  count_blobs(green_mask),   # FP
        "black_blobs":  count_blobs(black_mask)    # TN
    }

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------

# Paths
image_dir = "/..add your path/train_data/images/"
mask_dir = "/..add your path/u_net/train_data/masks/"

# List and filter image files
image_files = sorted([
    f for f in os.listdir(image_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
])
mask_files = sorted([
    f for f in os.listdir(mask_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
])

# Ensure equal number of images and masks
assert len(image_files) == len(mask_files), "Mismatch between the number of images and masks!"

# Pair images and masks
image_mask_pairs = list(zip(image_files, mask_files))

# First, split into train+validation (80%) and test (20%)
train_val_pairs, test_pairs = train_test_split(image_mask_pairs, test_size=0.2, random_state=42)

# From the train+val subset, allocate 25% to validation, leaving 75% for training.
# 25% of the 80% subset is 20% of the original dataset.
train_pairs, val_pairs = train_test_split(train_val_pairs, test_size=0.25, random_state=42)

# Unzip pairs
train_images, train_masks = zip(*train_pairs) if train_pairs else ([], [])
val_images, val_masks = zip(*val_pairs) if val_pairs else ([], [])
test_images, test_masks = zip(*test_pairs) if test_pairs else ([], [])

# Convert to lists
train_images, train_masks = list(train_images), list(train_masks)
val_images, val_masks = list(val_images), list(val_masks)
test_images, test_masks = list(test_images), list(test_masks)

assert set(train_images).isdisjoint(set(val_images)), "Data leakage detected between Train & Validation!"
assert set(train_images).isdisjoint(set(test_images)), "Data leakage detected between Train & Test!"


print(f"Training set: {len(train_images)} images")
print(f"Validation set: {len(val_images)} images")
print(f"Testing set: {len(test_images)} images")

# Augmentations
train_augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ElasticTransform(alpha=1, sigma=50, p=0.3),
    ToTensorV2()
])
val_transform = ToTensorV2()  # Typically just convert to tensor for val
test_transform = ToTensorV2() # Same for test

# Create datasets
train_dataset = SegmentationDataset(image_dir, mask_dir, train_images, train_masks, transform=train_augmentations, return_filename=True, return_orig_size=True)
val_dataset   = SegmentationDataset(image_dir, mask_dir, val_images,   val_masks,   transform=val_transform, return_filename=True ,return_orig_size=True)
test_dataset  = SegmentationDataset(image_dir, mask_dir, test_images,  test_masks,  transform=test_transform, return_filename=True ,return_orig_size=True)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

# -------------------------------
# Model Initialization
# -------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetDC(in_channels=3, out_channels=1)
model.to(device)
#logger.info("Model Summary:")
#summary(model, input_size=(3, 512, 512))

#print(model)

# -------------------------------
# Training Loop
# -------------------------------

num_epochs = 15
#criterion = combined_loss
criterion = lambda pred, tgt: focal_dice_loss(pred, tgt, alpha=1.0, gamma=2.0, ratio=0.3)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
scaler = torch.cuda.amp.GradScaler()  # Mixed precision


best_dice = 0.0
patience = 5
patience_counter = 0

# Metric lists
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
train_dices = []
val_dices = []

for epoch in range(num_epochs):
    # -------- TRAINING --------
    model.train()
    train_loss = 0.0
    train_dice_sum = 0.0
    train_correct = 0
    train_total = 0
    
    for images, masks  ,orig_size, filenames in train_loader:
        images, masks = images.float().to(device), masks.float().to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        predicted_masks = (outputs > 0.3).float().squeeze(1)
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        if predicted_masks.dim() == 3:
            predicted_masks = predicted_masks.unsqueeze(1)
        train_dice_sum += dice_coef(masks, predicted_masks).item()
        
        y_true_flat = masks.view(-1).cpu().numpy()
        y_pred_flat = predicted_masks.view(-1).cpu().numpy()
        train_correct += (y_true_flat == y_pred_flat).sum()
        train_total += len(y_true_flat)
        
    train_loss /= len(train_loader)
    train_dice_avg = train_dice_sum / len(train_loader)
    train_accuracy = train_correct / train_total
    
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    train_dices.append(train_dice_avg)

    # -------- VALIDATION --------
    model.eval()
    output_val_dir = "predicted_valDCfocal_masks"
    os.makedirs(output_val_dir, exist_ok=True)

    val_loss = 0.0
    val_dice_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for j, batch in enumerate(val_loader):
            if len(batch) == 4:  # (img, mask,orig_size, filename)
                images, masks,orig_size, filenames = batch
            else:
                images, masks = batch
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            predicted_masks = (outputs > 0.3).float()
            loss = criterion(outputs, masks)
            val_loss += loss.item()
        # Binarize predictions at 0.3 threshold
            predicted_masks = (outputs > 0.3).float()
        # Dice metric
            val_dice_sum += dice_coef(masks, predicted_masks).item()

        # Pixel-wise accuracy
            y_true_flat = masks.view(-1).cpu().numpy()
            y_pred_flat = predicted_masks.view(-1).cpu().numpy()
            val_correct += (y_true_flat == y_pred_flat).sum()
            val_total += len(y_true_flat)
        
        preds = (outputs > 0.3).float().cpu().numpy()  # Define the `preds` variable here

        for j in range(images.size(0)):
            pred_mask_256 = preds[j].squeeze()
            if isinstance(orig_size, list) and all(isinstance(t, torch.Tensor) for t in orig_size):
                orig_size = list(zip(orig_size[0].cpu().numpy(), orig_size[1].cpu().numpy()))
           
            sample_orig_h = int(orig_size[j][0])
            sample_orig_w = int(orig_size[j][1])

            # e.g. using OpenCV (nearest-neighbor is common for masks):
            pred_mask_orig = cv2.resize(pred_mask_256, (sample_orig_w, sample_orig_h), interpolation=cv2.INTER_NEAREST)
            pred_mask_orig = (pred_mask_orig * 255).astype(np.uint8)

            # Save with the same filename as your target mask if you want
            base_name, _ = os.path.splitext(filenames[j])
            out_path = os.path.join("predicted_valDCfocal_masks", f"{base_name}_pred.png")
            Image.fromarray(pred_mask_orig).save(out_path)
            print(f"Saved predicted mask: {out_path}")

    # Compute averages after the loop
    val_loss /= len(val_loader)
    val_dice_avg = val_dice_sum / len(val_loader)
    val_accuracy = val_correct / val_total

    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    val_dices.append(val_dice_avg)

    print(f"Epoch {epoch+1}/{num_epochs} | "
      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
      f"Train Dice: {train_dice_avg:.4f}, Val Dice: {val_dice_avg:.4f}")
    print(f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
    print("-------------------------------------------------------")

    # Early stopping check
    if val_dice_avg > best_dice:
        best_dice = val_dice_avg
        patience_counter = 0
        torch.save(model.state_dict(), "best_UNetDC_focal_model.pth")
        print("Model saved!")
    else:
        patience_counter += 1
    if patience_counter >= patience:
            print("Early stopping!")
            break


# -------------------------------
# Final Testing + Visualization
# -------------------------------
# After training + validation + model loading
model.load_state_dict(torch.load("best_UNetDC_focal_model.pth"))
model.eval()

test_loss     = 0.0
test_dice_sum = 0.0
test_correct  = 0
test_total    = 0
all_y_true    = []
all_y_pred    = []

# Create directories for saving
output_test_dir = "predicted_testDCfocal_masks"
os.makedirs(output_test_dir, exist_ok=True)
diff_map_dir    = "differences_map_test"
os.makedirs(diff_map_dir, exist_ok=True)
overlay_dir     = "overlay_diff_test"
os.makedirs(overlay_dir, exist_ok=True)

with torch.no_grad():
    for j, (images, masks, orig_size, filenames) in enumerate(test_loader):
        images, masks = images.to(device), masks.to(device)
        outputs       = model(images)

        # 1) Compute test loss
        loss = criterion(outputs, masks)
        test_loss += loss.item()

        # 2) Dice
        predicted_masks = (outputs > 0.3).float()
        test_dice_sum  += dice_coef(masks, predicted_masks).item()

        # 3) Pixel-wise accuracy
        y_true_flat = masks.view(-1).cpu().numpy()
        y_pred_flat = predicted_masks.view(-1).cpu().numpy()
        test_correct += (y_true_flat == y_pred_flat).sum()
        test_total   += len(y_true_flat)

        # For confusion matrix later
        all_y_true.append(y_true_flat)
        all_y_pred.append(y_pred_flat)

        # 4) Save predicted masks + difference maps
        preds_np  = predicted_masks.cpu().numpy()  # shape [B,1,H,W]
        true_np   = masks.cpu().numpy()            # shape [B,1,H,W]
        images_np = images.cpu().numpy()           # shape [B,3,H,W]

        for j in range(images.size(0)):
            # (a) Convert predicted mask to 2D
            pred_mask_2d = preds_np[j].squeeze()       # shape (H,W)
            true_mask_2d = true_np[j].squeeze()        # shape (H,W)

            # (b) Resize to original dims if needed
            if isinstance(orig_size, list) and all(isinstance(t, torch.Tensor) for t in orig_size):
                orig_size = list(zip(orig_size[0].cpu().numpy(), orig_size[1].cpu().numpy()))
            oh, ow = int(orig_size[j][0]), int(orig_size[j][1])

            if (oh, ow) != pred_mask_2d.shape[::-1]:
                pred_mask_2d = cv2.resize(pred_mask_2d, (ow, oh), interpolation=cv2.INTER_NEAREST)
                true_mask_2d = cv2.resize(true_mask_2d, (ow, oh), interpolation=cv2.INTER_NEAREST)

            # (c) Save raw predicted mask
            raw_pred_255 = (pred_mask_2d * 255).astype(np.uint8)
            base_name, _ = os.path.splitext(filenames[j])
            save_path = os.path.join(output_test_dir, f"{base_name}_pred.png")
            Image.fromarray(raw_pred_255).save(save_path)
            print(f"Saved predicted mask: {save_path}")

            # (d) Build difference map
            diff_map = create_difference_map(true_mask_2d, pred_mask_2d)
            diff_map_path = os.path.join(diff_map_dir, f"{base_name}_diffmap.png")
            Image.fromarray(diff_map).save(diff_map_path)
            print(f"Saved difference map: {diff_map_path}")

            # (e) Overlay difference on the original image
            img_3ch = images_np[j].transpose(1,2,0)  # shape (H,W,3), in [0,1]
            img_3ch = (img_3ch * 255).astype(np.uint8)

            # Possibly resize original as well
            if (oh, ow) != (img_3ch.shape[0], img_3ch.shape[1]):
                img_3ch = cv2.resize(img_3ch, (ow, oh), interpolation=cv2.INTER_LINEAR)

            overlayed   = overlay_difference(img_3ch, diff_map)
            overlay_path = os.path.join(overlay_dir, f"{base_name}_overlay.png")
            Image.fromarray(overlayed).save(overlay_path)
            print(f"Saved overlay difference: {overlay_path}")

# Summaries
test_loss    /= len(test_loader)
test_dice_avg = test_dice_sum / len(test_loader)
test_accuracy = test_correct / test_total

overall_test_accuracy = accuracy_score(np.concatenate(all_y_true), np.concatenate(all_y_pred))

print("========== Test Results ==========")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Dice: {test_dice_avg:.4f}")
print(f"Test Accuracy (pixel-wise): {test_accuracy:.4f}")
print(f"Test Accuracy (sklearn): {overall_test_accuracy:.4f}")

precision, recall, f1, specificity, conf_matrix = calculate_metrics(
    torch.tensor(np.concatenate(all_y_true)),
    torch.tensor(np.concatenate(all_y_pred))
)
plot_binary_confusion_matrix_with_metrics(conf_matrix, overall_test_accuracy)

######################################################################
# Additional Plots: Training vs. Validation

epochs_range = range(1, len(train_losses) + 1)

# Plot Loss & Dice
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Training Loss', color='red')
plt.plot(epochs_range, val_losses[len(train_losses)*-1:], label='Validation Loss', color='green')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_dices, label='Training Dice & focal', color='red')
plt.plot(epochs_range, val_dices[len(train_dices)*-1:], label='Validation Dice', color='green')
plt.xlabel('Epoch')
plt.ylabel('Dice')
plt.legend()
plt.title('Training and Validation Dice_Focal')
plt.tight_layout()
plt.savefig("loss_and_dice_focal_plot.png")

# Plot Accuracy
plt.figure(figsize=(6, 4))
plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
plt.plot(epochs_range, val_accuracies,  label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.tight_layout()
plt.savefig("accuracy_plot.png")
plt.close()

####################################################################
# Quick Visualization of Some Test Predictions
####################################################################
output_dir = "predicted_masks_test"
os.makedirs(output_dir, exist_ok=True)

with torch.no_grad():
    for j, (images, masks, orig_siz, mask_filenames) in enumerate(test_loader):
        images = images.to(device)
        outputs = model(images)
        # Binarize predictions at threshold=0.3
        predicted_masks = (outputs > 0.3).float().cpu().numpy()

        for j in range(images.size(0)):
            # Convert single predicted mask to 2D numpy array of shape (H, W)
            pred_mask_2d = predicted_masks[j].squeeze()

            # Resize to original image dimensions if desired
            sample_orig_h = int(orig_siz[j][0])
            sample_orig_w = int(orig_siz[j][1])
            pred_mask_orig = cv2.resize(
                pred_mask_2d, (sample_orig_w, sample_orig_h), interpolation=cv2.INTER_NEAREST
            )
            pred_mask_orig = (pred_mask_orig * 255).astype(np.uint8)

            # Construct the save path and write the final mask
            original_filename = mask_filenames[j]
            base_name, ext = os.path.splitext(original_filename)
            save_path = os.path.join(output_dir, f"{base_name}_pred.png")
            Image.fromarray(pred_mask_orig).save(save_path)

            print(f"Saved predicted mask to: {save_path}")



        # For demonstration, let's visualize up to 3 images from the batch
        for i in range(min(3, images.size(0))):
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(images[i].cpu().permute(1, 2, 0))
            plt.title("Original Image")
            plt.subplot(1, 3, 2)
            plt.imshow(masks[i].squeeze(), cmap="gray")
            plt.title("True Mask")
            plt.subplot(1, 3, 3)
            plt.imshow(predicted_masks[j].squeeze(), cmap="gray")
            plt.title("Predicted Mask")
            plt.savefig(f"prediction_visualization_test_batch{j}_img{j}.png")
            plt.close()

with torch.no_grad():
    for j, (images, masks, orig_siz, mask_filenames) in enumerate(test_loader):
        images = images.to(device)
        outputs = model(images)
        predicted_masks = (outputs > 0.3).float().cpu().numpy()  # shape [B,1,H,W]
        true_masks = masks.cpu().numpy()                         # shape [B,1,H,W]

        for j in range(images.size(0)):
            # Convert torch image to 3D CPU Numpy (H,W,3)
            img_3ch = images[j].cpu().permute(1,2,0).numpy()  # float in [0,1]
            img_3ch = (img_3ch * 255).astype(np.uint8)         # now uint8

            # Squeeze mask to 2D
            pred_mask_2d = predicted_masks[j].squeeze()
            true_mask_2d = true_masks[j].squeeze()

            # Build difference map
            diff_map = create_difference_map(true_mask_2d, pred_mask_2d)

            
            # Optionally, resize to original if needed
            # (assuming orig_siz is a list of (height, width))
            oh, ow = int(orig_siz[j][0]), int(orig_siz[j][1])
            if (oh, ow) != pred_mask_2d.shape[::-1]:
                # The predicted mask is 512x512 or so, so let's
                # resize 'diff_map' to original dimensions:
                diff_map = cv2.resize(diff_map, (ow, oh), interpolation=cv2.INTER_NEAREST)
                # Also resize 'img_3ch' if it's not the original size
                img_3ch = cv2.resize(img_3ch, (ow, oh), interpolation=cv2.INTER_LINEAR)

            # Now overlay the difference map on the original
            overlayed = overlay_difference(img_3ch, diff_map)

            # Save or show the result
            base_name, _ = os.path.splitext(mask_filenames[j])
            out_path = f"differences_overlay_batch{j}_img{j}.png"

            plt.figure(figsize=(12, 6))
            plt.subplot(1,3,1)
            plt.title("Original")
            plt.imshow(img_3ch)
            plt.axis("off")

            plt.subplot(1,3,2)
            plt.title("Diff Map")
            plt.imshow(diff_map)
            plt.axis("off")

            plt.subplot(1,3,3)
            plt.title("Overlayed")
            plt.imshow(overlayed)
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()

if __name__ == "__main__":
    pass


