import torch
import torch.nn as nn
import configparser
import matplotlib.pyplot as plt
import os
from datetime import datetime
from torchvision.datasets import ImageFolder
from ultralytics import YOLO
from huggingface_hub import hf_hub_download


# Disabling Weights & Biases logging
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'

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

# Make metrics file
metrics = results.results_dict
with open('metrics.txt', 'w') as f:
    f.write('Training Metrics:\n')
    f.write('-' * 20 + '\n\n')
    
    for epoch, values in enumerate(metrics.get('metrics', [])):
        f.write(f'Epoch {epoch + 1}:\n')
        for metric_name, value in values.items():
            f.write(f'{metric_name}: {value:.4f}\n')
        f.write('\n')
        
    # Save final metrics
    f.write('Final Results:\n')
    f.write('-' * 20 + '\n')
    for key, value in metrics.items():
        if key != 'metrics':
            f.write(f'{key}: {value}\n')
            
# Save model
save_dir = os.path.join(os.getcwd(), 'models')
os.makedirs(save_dir, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_save_path = os.path.join(save_dir, f'model_{timestamp}.pt')
model.save(model_save_path)
print(f"Model saved to: {model_save_path}")