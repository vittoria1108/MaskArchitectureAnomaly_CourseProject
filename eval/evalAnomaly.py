# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import glob
import torch
import random
from PIL import Image
import numpy as np
import os.path as osp
from argparse import ArgumentParser
from torch.amp.autocast_mode import autocast
from ood_metrics import fpr_at_95_tpr
from sklearn.metrics import average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch.nn.functional as F
import matplotlib.pyplot as plt

from erfnet import ERFNet
from eomt.models.vit import ViT
from eomt.models.eomt import EoMT

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Functions to calculate anomaly scores

def calculate_msp(logits, temperature=1.0):
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=0)
    max_probs, _ = torch.max(probs, dim=0)
    return (1.0 - max_probs).cpu().numpy()

def calculate_max_logit(logits):
    max_logits, _ = torch.max(logits, dim=0)
    return (-max_logits).cpu().numpy()

def calculate_entropy(logits):
    probs = F.softmax(logits, dim=0)
    log_probs = torch.log(probs + 1e-7)
    return (-torch.sum(probs * log_probs, dim=0)).cpu().numpy()

def calculate_rba(logits):
    return (-logits.tanh().sum(dim=0)).cpu().numpy()

def main():

    parser = ArgumentParser()

    # Validation dataset
    parser.add_argument(
        "--input",
        default="/content/drive/MyDrive/MaskArchitectureAnomaly_CourseProject/dataset/fs_static/*.jpg",
        nargs="+",
        help="Path to the validation images"
    )

    # Model weights
    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth", help="Name of the weights file (e.g., erfnet_pretrained.pth or eomt_cityscapes.bin)")
    
    # Models
    parser.add_argument("--model_type", type=str, default="eomt", choices=["erfnet", "eomt"], help="Choose which model to evaluate: erfnet or eomt")
    
    # CPU/GPU
    parser.add_argument('--cpu', action='store_true')

    # Arguments for logit normalization
    parser.add_argument('--apply_norm', action='store_true', help="Logit normalization flag (use it only if the model was trained with logit normalization)")
    parser.add_argument('--tau', type=float, default=0.04, help="Temperature used for logit normalization")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device in uso: {device} \n")

    # Images dimension based on the model type
    img_size = (1024, 1024) if args.model_type == "eomt" else (512, 1024)

    if args.model_type == "eomt":
        input_transform = Compose([
            Resize(img_size, Image.BILINEAR),
            ToTensor(),
        ])
    else:
        input_transform = Compose([Resize(img_size, Image.BILINEAR), ToTensor()])

    target_transform = Compose([Resize(img_size, Image.NEAREST)])

    print(f"START OF EVALUATION WITH MODEL: {args.model_type.upper()} \n")

    # Choice of model and loading weights
    if args.model_type == 'eomt':

        encoder = ViT(img_size=img_size, patch_size=16, backbone_name="vit_base_patch14_reg4_dinov2") 
        model = EoMT(encoder=encoder, num_classes=19, num_q=100, num_blocks=3, masked_attn_enabled=False) 

        weightspath = os.path.join(args.loadDir, args.loadWeights)
        print(f"\nWeights path for EoMT: {weightspath}")

        state_dict_raw = torch.load(weightspath, map_location='cpu', weights_only=False)

        # Weights extraction
        if 'state_dict' in state_dict_raw:
            state_dict = state_dict_raw['state_dict']
        else:
            state_dict = state_dict_raw

        # Cleaning of all possible spurious prefixes
        clean_state_dict = {}
        for key, value in state_dict.items():
            # If the key belongs to the criterion (training loss), we skip it
            if 'criterion' in key:
                continue
                
            # Removal of 'network.' and 'module.' 
            new_key = key.replace('network.', '').replace('module.', '')
            clean_state_dict[new_key] = value

        # Try/except to load the weights and avoid silent errors
        try:
            model.load_state_dict(clean_state_dict, strict=True)
            print("Checkpoint EoMT correctly loaded (Strict=True)")
        except Exception as e:
            print("\n" + "!"*60)
            print("ERROR: Checkpoint EoMT NOT loaded correctly.")
            print(e)
            print("!"*60 + "\n")
            sys.exit(1)
            
        model = model.to(device)

    elif args.model_type == 'erfnet':

        model = ERFNet(20)
        if not args.cpu:
            model = torch.nn.DataParallel(model).cuda()
            
        weightspath = os.path.join(args.loadDir, args.loadWeights)
        state_dict = torch.load(weightspath, map_location=lambda storage, loc: storage)
        
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                own_state[name].copy_(param)
            elif name.startswith("module.") and name.split("module.")[-1] in own_state:
                own_state[name.split("module.")[-1]].copy_(param)

        model = model.to(device)

    model.eval()

    # EVALUATION OF ANOMALY DETECTION

    print("\nAnomaly dataset evaluation in progress...")
    input_pattern_anom = os.path.expanduser(str(args.input[0]))
    files_anom = glob.glob(input_pattern_anom)

    # Sorting the images
    files_anom.sort()

    print(f"Found {len(files_anom)} anomaly images.")

    label_pattern_anom = os.path.expanduser(str(args.label[0]))
    labels_anom = glob.glob(label_pattern_anom)
    
    # Sorting the labels
    labels_anom.sort()

    # Lists will store only the useful pixels
    val_labels_list = []
    val_msp_list = []
    val_logit_list = []
    val_entropy_list = []
    val_rba_list = []
    
    # Temperature values for grid search
    t_values = [0.1, 0.25, 0.5, 0.75, 0.8, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 4.0,
                 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0]

    val_temp_list = {T: [] for T in t_values}

    for path in files_anom:
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().to(device)

        with torch.no_grad():

            # Choice of model type for anomaly score calculation
            if args.model_type == 'eomt':
                with autocast(device_type=device.type, dtype=torch.float16):

                    height_img, width_img = images.shape[-2], images.shape[-1]
                    mask_logits_per_layer, class_logits_per_layer = model(images)

                    mask_logits = F.interpolate(mask_logits_per_layer[-1], size=(height_img, width_img), mode="bilinear")
                    class_logits = class_logits_per_layer[-1]

                    mask_probs = mask_logits.sigmoid()
                    class_probs = F.softmax(class_logits, dim=-1)[..., :-1]
                    sem_seg_probs = torch.einsum("bqc, bqhw -> bchw", class_probs, mask_probs)

                    pixel_logits = torch.log(sem_seg_probs[0].float() + 1e-7)
                    
                    msp_score = calculate_msp(sem_seg_probs[0])
                    entropy_score = calculate_entropy(sem_seg_probs[0])
                    rba_score = calculate_rba(pixel_logits)

            elif args.model_type == 'erfnet':
                result = model(images)
                pixel_logits = result.squeeze(0) 

                msp_score = calculate_msp(pixel_logits)
                entropy_score = calculate_entropy(pixel_logits)
                rba_score = None              
            
            logit_score = calculate_max_logit(pixel_logits)

            # Temperature scaling for MSP scores with EoMT
            if not args.apply_norm and args.model_type == 'eomt':
                msp_t_scores_img = {}
                for T in t_values:
                    class_probs_T = F.softmax(class_logits / T, dim=-1)[..., :-1]
                    sem_seg_probs_T = torch.einsum("bqc, bqhw -> bchw", class_probs_T.float(), mask_probs.float())
                    msp_t_scores_img[T] = (1.0 - torch.max(sem_seg_probs_T[0], dim=0)[0]).cpu().numpy()

        # Ground Truth management
        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT: pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT: pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT: pathGT = pathGT.replace("jpg", "png")  

        if not os.path.exists(pathGT):
            continue

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)

        # Classes remapping for different datasets to unify the anomaly class as 1 and background as 0
        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)
        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        # Prediction visualization for debugging purposes
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(pathGT)))

        debug_filename = f"debug_pred_{dataset_name}.png"

        # Save the debug image only if it doesn't already exist
        if not os.path.exists(debug_filename):

            # Find the predicted class for each pixel
            pred_class = torch.argmax(pixel_logits, dim=0).cpu().numpy()
                    
            plt.figure(figsize=(18, 6))
                    
            # Original input image
            plt.subplot(1, 3, 1)
            plt.title("Input Image")
            img_vis = images[0].permute(1, 2, 0).cpu().numpy()
            img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-5)
            plt.imshow(img_vis)
            plt.axis('off')

            # Model's predicted classes
            plt.subplot(1, 3, 2)
            plt.title(f"Model's Predictions ({dataset_name})")
            plt.imshow(pred_class, cmap='tab20')
            plt.axis('off')

            # Mask
            plt.subplot(1, 3, 3)
            plt.title("Anomaly Ground Truth")
            plt.imshow(ood_gts, cmap='gray')
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(debug_filename)
            plt.close()
            print(f"\n[*] Debug image saved: {debug_filename}")

        if 1 in np.unique(ood_gts):
            gt_flat = ood_gts.flatten()
            
            # Filtering only the pixels of interest (0 and 1) to avoid including ignored pixels (255)
            mask_v = (gt_flat == 0) | (gt_flat == 1)
            
            if mask_v.any():
                val_labels_list.append(gt_flat[mask_v].astype(np.int8))
                val_msp_list.append(msp_score.flatten()[mask_v].astype(np.float32))
                val_logit_list.append(logit_score.flatten()[mask_v].astype(np.float32))
                val_entropy_list.append(entropy_score.flatten()[mask_v].astype(np.float32))
                
                if args.model_type == 'eomt':
                    val_rba_list.append(rba_score.flatten()[mask_v].astype(np.float32))
                    
                if not args.apply_norm:
                    for T in t_values:
                        val_temp_list[T].append(msp_t_scores_img[T].flatten()[mask_v].astype(np.float32))

        del images, pixel_logits
        torch.cuda.empty_cache()

    # Final metrics calculation

    print("\n" + "="*50)
    print("Final metrics calculation")
    print("="*50)

    val_label = np.concatenate(val_labels_list)
    del val_labels_list

    metrics = {
        "MSP" : val_msp_list,
        "MAX LOGIT": val_logit_list,
        "ENTROPIA": val_entropy_list
    }
    if args.model_type == 'eomt':
        metrics["RBA"] = val_rba_list

    with open('results.txt', 'a') as file:
        file.write(f"\n{'#'*40}\nREPORT {args.model_type.upper()}\n{'#'*40}\n")
        
        for name, chunks in metrics.items():
            val_out = np.concatenate(chunks)
            metrics[name] = None
            
            prc_auc = average_precision_score(val_label, val_out)
            fpr = fpr_at_95_tpr(val_out, val_label)

            print(f"[{name}] AUPRC: {prc_auc*100.0:.2f}% | FPR95: {fpr*100.0:.2f}%")
            file.write(f"[{name}] AUPRC: {prc_auc*100.0:.2f} | FPR95: {fpr*100.0:.2f}\n")
            del val_out

        if not args.apply_norm:
            print("\nTEST TEMPERATURE SCALING FOR MSP (GRID SEARCH)")
            print(f"{'Temp':<8} | {'AUPRC (%)':<12} | {'FPR95 (%)':<12}")
            file.write("\RESULTS MSP WITH TEMPERATURE SCALING:\n")

            for T in t_values:
                val_out_t = np.concatenate(val_temp_list[T])
                val_temp_list[T] = None
                
                prc_auc = average_precision_score(val_label, val_out_t)
                fpr = fpr_at_95_tpr(val_out_t, val_label)
                
                tipo = "(Standard)" if T == 1.0 else ""
                print(f"{T:<8.1f} | {prc_auc*100.0:<12.2f} | {fpr*100.0:<12.2f} {tipo}")
                file.write(f"T={T:.1f} -> AUPRC: {prc_auc*100.0:.2f} | FPR95: {fpr*100.0:.2f}\n")
                del val_out_t

    print("\nComplete report saved in 'results.txt'")

if __name__ == '__main__':
    main()