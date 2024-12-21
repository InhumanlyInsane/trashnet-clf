import os, cv2, torch
import albumentations as A
import numpy as np
from tqdm.auto import tqdm
from torchvision.transforms import transforms
from PIL import Image

def get_class_counts(dataset_path):
    """Get number of images in each class directory"""
    class_counts = {}
    for class_dir in os.listdir(dataset_path):
        if os.path.isdir(os.path.join(dataset_path, class_dir)):
            class_path = os.path.join(dataset_path, class_dir)
            images = [f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            class_counts[class_dir] = len(images)
    return class_counts

def augment_dataset_balanced(dataset_path):
    """Augment dataset to balance all classes"""
    class_counts = get_class_counts(dataset_path)
    
    # Find max count of images in a class
    trash_counts = {k:v for k,v in class_counts.items()}
    target_count = max(trash_counts.values())
    
    # Calculate needed augmentations per image for each class
    augmentations_needed = {}
    for class_name, count in trash_counts.items():
        if count < target_count:
            augmentations_per_image = int(np.ceil((target_count - count) / count))
            augmentations_needed[class_name] = augmentations_per_image
    
    # Define augmentation pipeline
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.3),
        A.RandomScale(scale_limit=0.2, p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.CLAHE(p=1.0)
    ])

    # Process each class directory
    for class_dir, augs_needed in augmentations_needed.items():
        if augs_needed == 0:
            continue
            
        class_path = os.path.join(dataset_path, class_dir)
        images = [f for f in os.listdir(class_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]

        print(f"Augmenting {class_dir} class images {augs_needed} times each...")
        for img_name in tqdm(images):
            img_path = os.path.join(class_path, img_name)
            
            # Read image
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Create augmented versions
            for i in range(augs_needed):
                augmented = transform(image=image)
                aug_image = augmented['image']
                aug_image = cv2.resize(aug_image, (512, 384))
                
                name, ext = os.path.splitext(img_name)
                new_name = f"{name}_aug_{i+1}{ext}"
                save_path = os.path.join(class_path, new_name)
                
                aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, aug_image_bgr)

    print("Balanced data augmentation completed!")

def apply_clahe(directory):
    """Apply CLAHE to all images in directory and subdirectories"""
    import cv2
    import os
    from tqdm import tqdm
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    for class_dir in os.listdir(directory):
        class_path = os.path.join(directory, class_dir)
        if os.path.isdir(class_path):
            print(f"Applying CLAHE to {class_dir} images...")
            for img_name in tqdm(os.listdir(class_path)):
                img_path = os.path.join(class_path, img_name)
                
                # Read and convert image
                img = cv2.imread(img_path)
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel
                cl = clahe.apply(l)
                
                # Merge channels and convert back
                limg = cv2.merge((cl, a, b))
                enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                
                # Save enhanced image
                cv2.imwrite(img_path, enhanced)