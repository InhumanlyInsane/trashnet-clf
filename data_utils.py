import os, cv2
import albumentations as A
import numpy as np
from tqdm.auto import tqdm

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
    """Augment dataset to balance all classes except trash"""
    class_counts = get_class_counts(dataset_path)
    
    # Find max count excluding trash class
    non_trash_counts = {k:v for k,v in class_counts.items() if k != 'trash'}
    target_count = max(non_trash_counts.values())
    
    # Calculate needed augmentations per image for each class
    augmentations_needed = {}
    for class_name, count in non_trash_counts.items():
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
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5)
    ])

    # Process each class directory
    for class_dir, augs_needed in augmentations_needed.items():
        if augs_needed == 0:
            continue
            
        class_path = os.path.join(dataset_path, class_dir)
        images = [f for f in os.listdir(class_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

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