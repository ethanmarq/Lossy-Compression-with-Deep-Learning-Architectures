#
#
#
import torch
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def save_images_with_colormap(image, pred_mask, orig_mask, output_dir, idx):
    """
    Save the original image, original mask, and predicted mask as color-mapped images.
    """
    os.makedirs(output_dir, exist_ok=True) # Directory For Images
    
    # Convert integer masks to color using the 'rainbow' colormap
    cmap = plt.get_cmap('rainbow')
    norm = mcolors.Normalize(vmin=0, vmax=3)  # 4 Classes

    pred_mask_colored = cmap(norm(pred_mask))[:, :, :3]  # Drop the alpha channel (Transparency)
    orig_mask_colored = cmap(norm(orig_mask))[:, :, :3]

    # Convert the color maps to PIL Images (Numpy Arrays to Images)
    pred_mask_img = Image.fromarray((pred_mask_colored * 255).astype(np.uint8))
    orig_mask_img = Image.fromarray((orig_mask_colored * 255).astype(np.uint8))

    # Save images
    image.save(os.path.join(output_dir, f"image_{idx}.png"))
    pred_mask_img.save(os.path.join(output_dir, f"predicted_mask_{idx}.png"))
    orig_mask_img.save(os.path.join(output_dir, f"original_mask_{idx}.png"))

def make_predictions_and_save(model, dataLoader, output_dir, num_images=5):
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataLoader)): #Enumerate adds a counter to a iterable. idx counts up while batch is the data from DataLoader
            if idx >= num_images:
                break  # Stop after saving num_images pairs (5)

            X = batch['data'].to('cuda')
            y = batch['label'].to('cuda')

            toPilImage = transforms.ToPILImage()
            image = toPilImage(X.squeeze().to('cpu'))
            
            origMask = y.squeeze().to('cpu').numpy()

            predictions = model(X)
            
            if predictions.shape[-2:] != y.shape[-2:]:
                predictions = resize(predictions, size=y.shape[-2:])
                
            
            pred_labels = torch.argmax(predictions, dim=1).squeeze().to('cpu').numpy()

            # Save the images with colormap applied
            save_images_with_colormap(image, pred_labels, origMask, output_dir, idx)


    
#
#
#
#
#

import argparse
import math
import os
import pathlib

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm


# LOCAL IMPORTS
def build_valid_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# CAV LOCAL IMPORT
class Compose:
    def __init__(self, transforms):
        # Transformations Object
        self.transforms = transforms
        

    def __call__(self, image, target):
        # Applies Transformations to all the images
        for t in self.transforms:
            image = t(image)
            target = t(target)
        # Converts to NpArray and then to Tensor
        image = transforms.ToTensor()(image)
        target = torch.tensor(np.array(target), dtype=torch.int64)
        return image, target

INPUT_IMAGE_HEIGHT = 1024
INPUT_IMAGE_WIDTH = 672 # 644 Original
# CONTINUE NON-LOCAL
    
    

from efficientvit.apps.utils import AverageMeter
from efficientvit.models.utils import resize

from efficientvit.seg_model_zoo import create_seg_model


# local datasets
# from datasets.rellis import RellisDataset
# from datasets.cityscapes import CityscapesDataset
from datasets.cav import SegmentationDataset



class SegIOU:
    def __init__(self, num_classes: int, ignore_index: int = -1) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = (outputs + 1) * (targets != self.ignore_index)
        targets = (targets + 1) * (targets != self.ignore_index)
        intersections = outputs * (outputs == targets)

        outputs = torch.histc(
            outputs,
            bins=self.num_classes,
            min=1,
            max=self.num_classes,
        )
        targets = torch.histc(
            targets,
            bins=self.num_classes,
            min=1,
            max=self.num_classes,
        )
        intersections = torch.histc(
            intersections,
            bins=self.num_classes,
            min=1,
            max=self.num_classes,
        )
        unions = outputs + targets - intersections

        return {
            "i": intersections,
            "u": unions,
        }


def get_canvas(
    image: np.ndarray,
    mask: np.ndarray,
    colors: tuple or list,
    opacity=0.5,
) -> np.ndarray:
    image_shape = image.shape[:2]
    mask_shape = mask.shape
    if image_shape != mask_shape:
        mask = cv2.resize(mask, dsize=(image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    seg_mask = np.zeros_like(image, dtype=np.uint8)
    for k, color in enumerate(colors):
        seg_mask[mask == k, :] = color
    canvas = seg_mask * opacity + image * (1 - opacity)
    canvas = np.asarray(canvas, dtype=np.uint8)
    return canvas


def main():
    parser = argparse.ArgumentParser()    
    
    # For CAV SegmentationDataset(root_dir, 'Test/Train')
    root_dirs = ['/scratch/marque6/CaTCompress']
    # root_dirs = 'CAT/mixed'
   # parser.add_argument("--path", type=str, default="/scratch/apicker/cityscapes/leftImg8bit/val")
   # parser.add_argument("--path", type=str, default="/scratch/apicker/rellis3d-nonfixed/test") 
    parser.add_argument("--path", type=str, default=root_dirs) #CAV
    parser.add_argument("--dataset", type=str, default="cav", choices=["cityscapes", "rellis", "cav"])
    parser.add_argument("--batch_size", help="batch size per gpu", type=int, default=1)
    parser.add_argument("-j", "--workers", help="number of workers", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=1024)
    parser.add_argument("--model", type=str, default='b0')
    parser.add_argument("--weight_url", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)

    args = parser.parse_args()

    if args.dataset == "cityscapes":
        dataset = CityscapesDataset(args.path, (args.crop_size, args.crop_size * 2))
        print(f'len of dataset: {len(dataset)}')
    elif args.dataset == "rellis":
        dataset = RellisDataset(args.path)
        dataset.transform = build_valid_transform()
        print(f'len of dataset: {len(dataset)}')
    elif args.dataset == "cav":
        transform = Compose([transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH), interpolation=Image.NEAREST)])
        dataset = SegmentationDataset(args.path, 'Test', transforms=transform)
        # dataset.transform = build_valid_transform()
        print(f'len of dataset: {len(dataset)}')
        
    else:
        raise NotImplementedError
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    
    print(f'Length of dataloader: {len(data_loader)}')

    model = create_seg_model(args.model, args.dataset, weight_url=args.weight_url)
       # for name, layer in model.named_modules():
     #   print(name, layer)
        
    print('weight layer data')
   # print(model.head.input_ops[0].op_list[0].conv.weight.data)
   # quit()

    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)

    interaction = AverageMeter()
    union = AverageMeter()
    iou = SegIOU(len(dataset.classes))

    num = 1
    with torch.inference_mode():
        with tqdm(total=len(data_loader), desc=f"Eval {args.model} on {args.dataset}") as t:
            for feed_dict in data_loader:
                images, mask = feed_dict["data"].cuda(), feed_dict["label"].cuda()
                
                # compute output
                output = model(images)
                # resize the output to match the shape of the mask
                if output.shape[-2:] != mask.shape[-2:]:
                    output = resize(output, size=mask.shape[-2:])
                output = torch.argmax(output, dim=1)
                if num == 1:
                    print(f'size of i,m,o (after argmax and upsample) {images.shape} {mask.shape} {output.shape}')
                    num = 0
                stats = iou(output, mask)
                
                interaction.update(stats["i"])
                union.update(stats["u"])

                
                
                t.set_postfix(
                    {
                        "mIOU": (interaction.sum / union.sum).cpu().mean().item() * 100,
                        "image_size": list(images.shape[-2:]),
                    }
                )
                t.update()

                if args.save_path is not None:
                    with open(os.path.join(args.save_path, "summary.txt"), "a") as fout:
                        for i, (idx, image_path) in enumerate(zip(feed_dict["index"], feed_dict["image_path"])):
                            pred = output[i].cpu().numpy()
                            raw_image = np.array(Image.open(image_path).convert("RGB"))
                            canvas = get_canvas(raw_image, pred, dataset.class_colors)
                            canvas = Image.fromarray(canvas)
                            canvas.save(os.path.join(args.save_path, f"{idx}.png"))
                            fout.write(f"{idx}:\t{image_path}\n")

    mIoU = (interaction.sum / union.sum).cpu().mean().item() * 100
    class_IoUs = (interaction.sum / union.sum).cpu().numpy() * 100
    print(f"mIoU = {mIoU:.3f}")
    print("Class IoUs:")
    for i, class_iou in enumerate(class_IoUs):
        print(f"Class {i}: {class_iou:.3f}")
        
    '''
    # Prints out 5 images, orignal masks, predicted masks
    # into /output_images to visulize the predicted data
    #
    root_dirs = ['/scratch/marque6/CaTCompress']
    transform = Compose([transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH), interpolation=Image.NEAREST)])
    dataset = SegmentationDataset(root_dirs, 'Test', transforms=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # Process one image at a time
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    
     output_dir = "output_images"
     make_predictions_and_save(model, data_loader, output_dir, num_images=5)
     #
     # Ends Here
     #
    '''

if __name__ == "__main__":
    main()
