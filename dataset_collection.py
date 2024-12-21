import os, random, shutil
from zipfile import ZipFile
from huggingface_hub import hf_hub_download
from data_utils import augment_dataset_balanced

if __name__ == "__main__":
    
    # Download only the dataset-resized.zip file
    file_path = hf_hub_download(
        repo_id="garythung/trashnet",
        filename="dataset-resized.zip",
        repo_type="dataset",
        local_dir="trashnet"
    )

    # Create train and validation directories
    os.makedirs('./data-main/train', exist_ok=True)
    os.makedirs('./data-main/val', exist_ok=True)
    # Create train and val directories for each class
    # Remove trash class => potential noise
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
    for class_name in classes:
        os.makedirs(f'./data-main/train/{class_name}', exist_ok=True)
        os.makedirs(f'./data-main/val/{class_name}', exist_ok=True)

    # Extract resized dataset from trashnet
    dataset_dir = './trashnet/dataset-resized.zip'
    with ZipFile(dataset_dir, 'r') as zip_ref:
        zip_ref.extractall('')
        
    # Split and move files for each class
    for class_name in classes:
        # Get all files in source directory
        src_dir = f'./dataset-resized/{class_name}'
        files = os.listdir(src_dir)

        # Randomly shuffle files
        random.shuffle(files)

        # Calculate split index
        split_idx = int(len(files) * 0.8)

        # Split into train and val
        train_files = files[:split_idx]
        val_files = files[split_idx:]

        # Move train files
        for f in train_files:
            src = os.path.join(src_dir, f)
            dst = f'./data-main/train/{class_name}/{f}'
            shutil.copy(src, dst)

        # Move val files
        for f in val_files:
            src = os.path.join(src_dir, f)
            dst = f'./data-main/val/{class_name}/{f}'
            shutil.copy(src, dst)
            
    # Remove extracted datasets
    shutil.rmtree('./dataset-resized')
    shutil.rmtree('./trashnet')
    shutil.rmtree('./__MACOSX')

    # Augment dataset to balance classes
    augment_dataset_balanced('./data-main/train')