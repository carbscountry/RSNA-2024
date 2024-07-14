import os
import gc
import sys
import time
from PIL import Image
import cv2
import math, random
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

import timm
from transformers import get_cosine_schedule_with_warmup

import albumentations as A

from sklearn.model_selection import KFold

import wandb

### set path 

if os.getcwd() == '/kaggle/working':
    rd = '/kaggle/'
    cvt_path = os.path.join(rd, '/cvt_png')
else:
    rd = '/workspace/'
    cvt_path = os.path.join(rd, 'output/cvt_png')
input_path = os.path.join(rd, 'input/rsna-2024-lumbar-spine-degenerative-classification') 

local_time = time.strftime('%Y-%m-%d-%H', time.localtime(time.time()))

### config 

NOT_DEBUG = False # True -> run naormally, False -> debug mode, with lesser computing cost

OUTPUT_DIR = os.path.join(rd,f'output/rsna24-results/{local_time}')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
N_WORKERS = os.cpu_count() 
USE_AMP = True # can change True if using T4 or newer than Ampere
SEED = 8620

IMG_SIZE = [512, 512]
IN_CHANS = 30
N_LABELS = 25
N_CLASSES = 3 * N_LABELS

AUG_PROB = 0.75

N_FOLDS = 5 if NOT_DEBUG else 2
EPOCHS = 20 if NOT_DEBUG else 2
MODEL_NAME = "tf_efficientnet_b3.ns_jft_in1k" if NOT_DEBUG else "tf_efficientnet_b0.ns_jft_in1k"

GRAD_ACC = 2
TGT_BATCH_SIZE = 32
BATCH_SIZE = TGT_BATCH_SIZE // GRAD_ACC
MAX_GRAD_NORM = None
EARLY_STOPPING_EPOCH = 3

LR = 2e-4 * TGT_BATCH_SIZE / 32
WD = 1e-2
AUG = True
os.makedirs(OUTPUT_DIR, exist_ok=True)
### set random seed

def set_random_seed(seed: int = 8620, deterministic: bool = False):
    """Set seeds"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = deterministic  # type: ignore
    
### define dataset
    
class RSNA24Dataset(Dataset):
    def __init__(self, df, phase='train', transform=None):
        self.df = df
        self.transform = transform
        self.phase = phase
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = np.zeros((512, 512, IN_CHANS), dtype=np.uint8)
        t = self.df.iloc[idx]
        st_id = int(t['study_id'])
        label = t[1:].values.astype(np.int64)
        
        # Sagittal T1
        for i in range(0, 10, 1):
            try:
                p = os.path.join(cvt_path,f'{st_id}/Sagittal T1/{i:03d}.png')
                img = Image.open(p).convert('L')
                img = np.array(img)
                x[..., i] = img.astype(np.uint8)
            except:
                #print(f'failed to load on {st_id}, Sagittal T1')
                pass
            
        # Sagittal T2/STIR
        for i in range(0, 10, 1):
            try:
                p = os.path.join(cvt_path,f'{st_id}/Sagittal T2_STIR/{i:03d}.png')
                img = Image.open(p).convert('L')
                img = np.array(img)
                x[..., i+10] = img.astype(np.uint8)
            except:
                #print(f'failed to load on {st_id}, Sagittal T2/STIR')
                pass
            
        # Axial T2
        axt2 = glob(os.path.join(cvt_path,f'{st_id}/Axial T2/*.png'))
        axt2 = sorted(axt2)
    
        step = len(axt2) / 10.0
        st = len(axt2)/2.0 - 4.0*step
        end = len(axt2)+0.0001
                
        for i, j in enumerate(np.arange(st, end, step)):
            try:
                p = axt2[max(0, int((j-0.5001).round()))]
                img = Image.open(p).convert('L')
                img = np.array(img)
                x[..., i+20] = img.astype(np.uint8)
            except:
                #print(f'failed to load on {st_id}, Sagittal T2/STIR')
                pass  
            
        assert np.sum(x)>0
            
        if self.transform is not None:
            x = self.transform(image=x)['image']

        x = x.transpose(2, 0, 1)
                
        return x, label
    
### define model

class RSNA24Model(nn.Module):
    def __init__(self, model_name, in_c=30, n_classes=75, pretrained=True, features_only=False):
        super().__init__()
        self.model = timm.create_model(
                                    model_name,
                                    pretrained=pretrained, 
                                    features_only=features_only,
                                    in_chans=in_c,
                                    num_classes=n_classes,
                                    global_pool='avg'
                                    )
    
    def forward(self, x):
        y = self.model(x)
        return y

### transform
transforms_train = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=AUG_PROB),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
        A.GaussNoise(var_limit=(5.0, 30.0)),
    ], p=AUG_PROB),

    A.OneOf([
        A.OpticalDistortion(distort_limit=1.0),
        A.GridDistortion(num_steps=5, distort_limit=1.),
        A.ElasticTransform(alpha=3),
    ], p=AUG_PROB),

    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=AUG_PROB),
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.CoarseDropout(max_holes=16, max_height=64, max_width=64, min_holes=1, min_height=8, min_width=8, p=AUG_PROB),    
    A.Normalize(mean=0.5, std=0.5)
])

transforms_val = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.Normalize(mean=0.5, std=0.5)
])

if not NOT_DEBUG or not AUG:
    transforms_train = transforms_val


### set wandb
wandb.init(
    project='ITK_rsna24',
    config={
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'model_name': MODEL_NAME,
        'lr': LR,
        'wd': WD,
        'aug': AUG,
        'seed': SEED,
        'use_amp': USE_AMP,
        'max_grad_norm': MAX_GRAD_NORM,
        'grad_acc': GRAD_ACC,
        'early_stopping_epoch': EARLY_STOPPING_EPOCH,
        'n_folds': N_FOLDS,
        'img_size': IMG_SIZE,
        'in_chans': IN_CHANS,
        'n_labels': N_LABELS,
        'n_classes': N_CLASSES,
        'output_dir': OUTPUT_DIR,
        'device': device
    }
)
   
### train 
def trainer(df):
    
    autocast = torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.half)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP, init_scale=4096)
    
    skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(df)))):
        print('#'*30)
        print(f'start fold{fold}')
        print('#'*30)
        print(len(trn_idx), len(val_idx))
        df_train = df.iloc[trn_idx]
        df_valid = df.iloc[val_idx]

        train_ds = RSNA24Dataset(df_train, phase='train', transform=transforms_train)
        train_dl = DataLoader(
                    train_ds,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    pin_memory=True,
                    drop_last=True,
                    num_workers=N_WORKERS
                    )

        valid_ds = RSNA24Dataset(df_valid, phase='valid', transform=transforms_val)
        valid_dl = DataLoader(
                    valid_ds,
                    batch_size=BATCH_SIZE*2,
                    shuffle=False,
                    pin_memory=True,
                    drop_last=False,
                    num_workers=N_WORKERS
                    )

        model = RSNA24Model(MODEL_NAME, IN_CHANS, N_CLASSES, pretrained=True)
        model.to(device)
        
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)

        warmup_steps = EPOCHS/10 * len(train_dl) // GRAD_ACC
        num_total_steps = EPOCHS * len(train_dl) // GRAD_ACC
        num_cycles = 0.475
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_total_steps,
                                                    num_cycles=num_cycles)

        weights = torch.tensor([1.0, 2.0, 4.0])
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        criterion2 = nn.CrossEntropyLoss(weight=weights)

        best_loss = 1.2
        best_wll = 1.2
        es_step = 0

        for epoch in range(1, EPOCHS+1):
            print(f'start epoch {epoch}')
            model.train()
            total_loss = 0
            with tqdm(train_dl, leave=True) as pbar:
                optimizer.zero_grad()
                for idx, (x, t) in enumerate(pbar):  
                    x = x.to(device)
                    t = t.to(device)
                    
                    with autocast:
                        loss = 0
                        y = model(x)
                        for col in range(N_LABELS):
                            pred = y[:,col*3:col*3+3]
                            gt = t[:,col]
                            loss = loss + criterion(pred, gt) / N_LABELS
                            
                        total_loss += loss.item()
                        if GRAD_ACC > 1:
                            loss = loss / GRAD_ACC
        
                    if not math.isfinite(loss):
                        print(f"Loss is {loss}, stopping training")
                        sys.exit(1)
        
                    pbar.set_postfix(
                        OrderedDict(
                            loss=f'{loss.item()*GRAD_ACC:.6f}',
                            lr=f'{optimizer.param_groups[0]["lr"]:.3e}'
                        )
                    )
                    scaler.scale(loss).backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM or 1e9)
                    
                    if (idx + 1) % GRAD_ACC == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        if scheduler is not None:
                            scheduler.step()                    
        
            train_loss = total_loss/len(train_dl)
            print(f'train_loss:{train_loss:.6f}')

            total_loss = 0
            y_preds = []
            labels = []
            
            model.eval()
            with tqdm(valid_dl, leave=True) as pbar:
                with torch.no_grad():
                    for idx, (x, t) in enumerate(pbar):
                        
                        x = x.to(device)
                        t = t.to(device)
                            
                        with autocast:
                            loss = 0
                            loss_ema = 0
                            y = model(x)
                            for col in range(N_LABELS):
                                pred = y[:,col*3:col*3+3]
                                gt = t[:,col]

                                loss = loss + criterion(pred, gt) / N_LABELS
                                y_pred = pred.float()
                                y_preds.append(y_pred.cpu())
                                labels.append(gt.cpu())
                            
                            total_loss += loss.item()   
        
            val_loss = total_loss/len(valid_dl)
            
            y_preds = torch.cat(y_preds, dim=0)
            labels = torch.cat(labels)
            val_wll = criterion2(y_preds, labels)
            
            print(f'val_loss:{val_loss:.6f}, val_wll:{val_wll:.6f}')
            wandb.log({'fold':fold, 'epoch':epoch, 'train_loss':train_loss, 'val_loss':val_loss, 'val_wll':val_wll})

            if val_loss < best_loss or val_wll < best_wll:
                
                es_step = 0

                if device!='cuda:0':
                    model.to('cuda:0')                
                    
                if val_loss < best_loss:
                    print(f'epoch:{epoch}, best loss updated from {best_loss:.6f} to {val_loss:.6f}')
                    best_loss = val_loss
                    
                if val_wll < best_wll:
                    print(f'epoch:{epoch}, best wll_metric updated from {best_wll:.6f} to {val_wll:.6f}')
                    best_wll = val_wll
                    fname = os.path.join(OUTPUT_DIR,f'best_wll_model_fold-{fold}.pt')
                    torch.save(model.state_dict(), fname)
                
                if device!='cuda:0':
                    model.to(device)
                
            else:
                es_step += 1
                if es_step >= EARLY_STOPPING_EPOCH:
                    print('early stopping')
                    break
    wandb.finish()