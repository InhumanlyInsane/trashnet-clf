import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import configparser
import os

os.environ['WANDB_DISABLED'] = 'true'

def load_config(config_path='train.cfg'):
    config = configparser.ConfigParser()
    config.read(config_path)
    return {
        'epochs': config.getint('Training', 'epochs', fallback=50),
        'lr': config.getfloat('Training', 'lr', fallback=0.001),
        'weight_decay': config.getfloat('Training', 'weight_decay', fallback=0.0005),
        'warmup_epochs': config.getint('Training', 'warmup_epochs', fallback=3),
        'momentum': config.getfloat('Training', 'momentum', fallback=0.937),
        'batch_size': config.getint('Training', 'batch_size', fallback=32),
        'device': config.get('Training', 'device', fallback='cuda' if torch.cuda.is_available() else 'cpu')
    }

# Load configurations and model
config = load_config()
model_path = hf_hub_download(repo_id="SoyoKaze83/trashnet-clf", filename="weights/yolov8.pt")
model = YOLO(model_path)

def prepare_data_yaml():
    data_yaml = """
path: data-main  # dataset root dir
train: train  # train images
val: val  # val images

# Classes
names:
    0: cardboard
    1: glass
    2: metal
    3: paper
    4: plastic
"""
    with open('data.yaml', 'w') as f:
        f.write(data_yaml)

# Load pre-trained model
model = YOLO(model_path)

# Prepare data configuration
# prepare_data_yaml()

data_path = os.path.join(os.getcwd(), 'data-main')
# Fine-tune the model
results = model.train(
    data=data_path,
    epochs=config['epochs'],
    batch=config['batch_size'],
    lr0=config['lr'],
    weight_decay=config['weight_decay'],
    warmup_epochs=config['warmup_epochs'],
    momentum=config['momentum'],
    device=config['device']
)