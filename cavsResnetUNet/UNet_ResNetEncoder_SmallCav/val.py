import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import Compose, SegmentationDataset  # Assuming you have a custom dataset class
import numpy as np
import segmentation_models_pytorch as smp
from sklearn.metrics import confusion_matrix
import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from PIL import Image


#training arguments passed for which compression dataset to train on
if len(sys.argv) > 1:
    compressor = sys.argv[1]
    #0 is base dataset no compression, 1 is 1E-1, 2 is 1E-2, etc up to 1E-7
    model_dir = "/home/aniemcz/cavsResnetUNet/UNet_ResNetEncoder_SmallCav/models/"
    save_path = f"/scratch/aniemcz/cavResnetUnet/{compressor}/"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    root_dir = "/scratch/aniemcz/cavsMiniLossyCompressorsV2/"
else:
    raise Exception("Need to pass the compressor name as arguments")


df = pd.DataFrame({
        'Model':[],
        'Compressor':[],
        'error_bound':[],
        'epochs':[],
        'miou':[],
        'class':[],
        'iou':[],
        'run':[]})

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

num_classes = 4

#nums = list(range(1, 101))
error_bound = ['1E-7','1E-6','1E-5','1E-4','1E-3','1E-2','1E-1']
#error_bound = [str(r) for r in nums]

# Data transformations
INPUT_IMAGE_HEIGHT = 672
INPUT_IMAGE_WIDTH = 1024
transform = Compose([transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH), interpolation=Image.NEAREST)])

model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=num_classes).to(device)

def validate_epoch(model, val_loader):
    def calculate_iou(pred, target):
        iou_per_class = []
        pred = torch.argmax(pred, dim=1)
        
        for cls in range(num_classes):
            pred_inds = (pred == cls)
            target_inds = (target == cls)
            
            intersection = (pred_inds & target_inds).sum().item()
            union = (pred_inds | target_inds).sum().item()
            
            if union == 0:
                iou_per_class.append(float('nan'))
            else:
                iou_per_class.append(intersection / union)
        
        return iou_per_class

    def mean_iou(iou_list):
        valid_iou = [iou for iou in iou_list if not np.isnan(iou)]
        if len(valid_iou) == 0:
            return float('nan')
        return np.mean(valid_iou)
    
    model.eval()
    iou_scores = []

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc='Validating'):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            # Calculate IoU
            iou = calculate_iou(outputs, masks)
            iou_scores.append(iou)

    # Compute mean IoU
    iou_scores = np.array(iou_scores)
    miou_per_class = np.nanmean(iou_scores, axis=0)
    miou = mean_iou(miou_per_class)

    print(f"Mean IoU: {miou}")  # Only print the mean IoU
    return miou, miou_per_class


for j, bound in tqdm(enumerate(error_bound)):
        model_path = os.path.join(model_dir, compressor, bound, 'model_epoch_120_aug.pth')
                            
        print(model_path)

        # Instantiate the model
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        test_imgs_dir = os.path.join(root_dir, f"{compressor}/{bound}/Test/rgb")
        test_masks_dir = os.path.join(root_dir, "mixed/Test/annos/int_maps")
        testDS = SegmentationDataset(imgs_dir=test_imgs_dir, masks_dir=test_masks_dir, transforms=transform)
        test_loader = DataLoader(testDS, batch_size=1, shuffle=False)
        
        mean_iou, iou_per_class = validate_epoch(model, test_loader)

        print("iou_per_class")

        for i in range(num_classes):
            print(iou_per_class[i])

        print(iou_per_class)

        print(f"Mean IoU: {mean_iou}")

        classes = ["Background","Sedan", "Pickup", "Offroad"]

        for i in range(num_classes):
            df.loc[len(df)] = ['unet', compressor, error_bound[j], '10', mean_iou, classes[i], iou_per_class[i], 1]

df.to_csv(
    os.path.join(save_path, f'UNET_train_{compressor}.csv')
)
    