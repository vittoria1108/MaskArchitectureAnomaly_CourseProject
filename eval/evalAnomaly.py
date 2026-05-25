# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
import os.path as osp
from argparse import ArgumentParser
from torch.amp.autocast_mode import autocast
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch.nn.functional as F
import iouEval

from erfnet import ERFNet
from eomt.models.vit import ViT
from eomt.models.eomt import EoMT

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        # Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)

# Funzioni per calcolo delle metriche

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

def calculate_rba(logits):
    return (-logits.tanh().sum(dim=0)).cpu().numpy()

# Matrice confusione per metrica miou
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def main():

    parser = ArgumentParser()
    
    # Immagini Cityscapes (FASE 1 - Normali)
    parser.add_argument(
        "--input_cityscapes",
        default="/content/cityscapes/leftImg8bit/val/*/*.png",
        help="Percorso delle immagini Cityscapes per la calibrazione"
    )

    # Immagini Anomalie (FASE 2 - OOD)
    parser.add_argument(
        "--input",
        default="/content/drive/MyDrive/MaskArchitectureAnomaly_CourseProject/dataset/fs_static/*.jpg",
        nargs="+",
        help="Percorso delle immagini da valutare"
    )
    
    # Pesi dei modelli
    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth", help="Nome file pesi (es. erfnet_pretrained.pth o eomt_cityscapes.bin)")
    
    # Modelli
    parser.add_argument("--model_type", type=str, default="eomt", choices=["erfnet", "eomt"], help="Scegli quale modello valutare: erfnet o eomt")
    
    # Usare la CPU
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device in uso: {device}")

    # Imposto dimensioni modello
    #img_size = (640, 640) if args.model_type == "eomt" else (512, 1024) CONTROLLARE, IN CONFIGS DICE 640 MA IL FILE .BIN CHIEDE 1024
    img_size = (1024, 1024) if args.model_type == "eomt" else (512, 1024)
    
    if args.model_type == "eomt":
        input_transform = Compose([
            Resize(img_size, Image.BILINEAR),
            ToTensor(),
            #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        input_transform = Compose([Resize(img_size, Image.BILINEAR), ToTensor()])

    target_transform = Compose([Resize(img_size, Image.NEAREST)])
    
    # Liste per risultati
    anomaly_score_msp_list = []
    anomaly_score_logit_list = []
    anomaly_score_entropy_list = []
    anomaly_score_rba_list = []
    ood_gts_list = []

    all_logits_for_eval = []
    all_labels_for_eval = []

    print(f"=== INIZIO VALUTAZIONE CON MODELLO: {args.model_type.upper()} ===")

    # Scelgo il modello da caricare
    if args.model_type == 'eomt':

        encoder = ViT(img_size=img_size, patch_size=16, backbone_name="vit_base_patch14_reg4_dinov2") 
        model = EoMT(encoder=encoder, num_classes=19, num_q=100, num_blocks=3, masked_attn_enabled=False) 

        weightspath = os.path.join(args.loadDir, args.loadWeights)
        print(f"\n Caricamento pesi EoMT da: {weightspath}")

        # Carichiamo lo state_dict nativo dal file .bin
        state_dict = torch.load(weightspath, map_location='cpu', weights_only=True)

        # Pulizia di tutti i possibili prefissi spuri
        clean_state_dict = {}
        for key, value in state_dict.items():
            # Se la chiave appartiene al criterion (loss di training), la saltiamo
            if 'criterion' in key:
                continue
                
            # Rimuoviamo sia 'network.' sia 'module.'
            new_key = key.replace('network.', '').replace('module.', '')
            clean_state_dict[new_key] = value

        # Try per assicurarci di aver letto il checkpoint correttamente
        try:
            model.load_state_dict(clean_state_dict, strict=True)
            print("Checkpoint EoMT caricato correttamente (Strict=True)")
        except Exception as e:
            print("\n" + "!"*60)
            print("ERRORE: Checkpoint EoMT NON caricato correttamente")
            print(e)
            print("!"*60 + "\n")
            sys.exit(1) # Blocca lo script immediatamente per evitare inferenze con pesi casuali
            
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

    # CALIBRAZIONE SU CITYSCAPES (IN-DISTRIBUTION)
    
    if args.model_type == 'eomt':

        print("\n--- FASE 1: Lettura Cityscapes (Immagini Normali ID) ---")
        input_pattern_city = os.path.expanduser(str(args.input_cityscapes))
        files_city = glob.glob(input_pattern_city)
        print(f"Trovate {len(files_city)} immagini Cityscapes.")

        # Matrice di confusione 19x19 per la mIoU
        #hist = np.zeros((19, 19))

        # Inizializziamo evaluator 20 classi totali (da 0 a 19)
        iouEvalVal = iouEval.iouEval(nClasses=20, ignoreIndex=19)

        for path in files_city:

            #print(f"Cityscapes ID: {os.path.basename(path)}")
            images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().to(device)

            with torch.no_grad():
                with autocast(dtype=torch.float16, device_type="cuda"):

                    # prende dim immagini, le fa passare a eomt, torna dove sono gli oggetti e che classi sono
                    altezza_img, larghezza_img = images.shape[-2], images.shape[-1]
                    mask_logits_per_layer, class_logits_per_layer = model(images)

                    # prende risultati dell'ultimo strato di rete, stira le maschere riportandole alla risoluzione originale
                    mask_logits = F.interpolate(mask_logits_per_layer[-1], size=(altezza_img, larghezza_img), mode="bilinear")
                    class_logits = class_logits_per_layer[-1] 

                    # le maschere vengono passate in una sigmoid (diventano probabilità tra 0 e 1)
                    # le classi passano in una softmax (diventano percentuali)
                    # [..., :-1] butta via l'ultimissima classe (lo "sfondo/void")
                    mask_probs = mask_logits.sigmoid()
                    class_probs = F.softmax(class_logits, dim=-1)[..., :-1]

                    # moltiplica le probabilità delle classi con le probabilità delle maschere, creando la mappa di segmentazione finale
                    sem_seg_probs = torch.einsum("bqc, bqhw -> bchw", class_probs, mask_probs)
                    
                    # Troviamo la classe vincente prevista dalla rete per calcolare la mIoU
                    #preds = torch.argmax(sem_seg_probs[0], dim=0).cpu().numpy()

                    #Estraiamo l'argmax mantenendo il formato [B, 1, H, W] richiesto da iouEval
                    preds = torch.argmax(sem_seg_probs, dim=1, keepdim=True)

            # Ground Truth di Cityscapes
            pathGT = path.replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png").replace("leftImg8bit", "gtFine")           
            # Se non esiste labelTrainIds, proviamo labelIds normale
            if not os.path.exists(pathGT):
                pathGT = path.replace("_leftImg8bit.png", "_gtFine_labelIds.png").replace("leftImg8bit", "gtFine")

            # apre gt e converte in array numeri
            if os.path.exists(pathGT):
                mask = Image.open(pathGT)
                mask = target_transform(mask)

                #gt_city = np.array(mask)

                # Aggiungiamo i dati alla matrice (velocissimo)
                #hist += fast_hist(gt_city.flatten(), preds.flatten(), 19)

                # Convertiamo la GT in un tensore PyTorch su GPU [B, 1, H, W]
                gt_tensor = torch.from_numpy(np.array(mask)).long().to(device).unsqueeze(0).unsqueeze(0)

                # Rimappiamo l'ignore label di Cityscapes da 255 a 19 per evitare crash
                gt_tensor[gt_tensor == 255] = 19

                # Passiamo i tensori (entrambi su GPU) direttamente a iouEval
                iouEvalVal.addBatch(preds, gt_tensor)

            del images, sem_seg_probs, preds
            torch.cuda.empty_cache()

        # Alla fine delle 500 immagini, calcoliamo la percentuale mIoU
        #iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        #miou_val = np.nanmean(iu)
        #print(f"Cityscapes mIoU: {miou_val * 100.0:.2f}%\n")   

        miou_val, iou_classes = iouEvalVal.getIoU()
        
        print(f"Cityscapes mIoU: {miou_val.item() * 100.0:.2f}%\n")   
        
        # per fare debugging
        print("IoU per classe (%):", (iou_classes.cpu().numpy() * 100))
            
    else: # modello = erfnet
        print("\n --- Il modello è erfnet, calibrazione su Cityscapes ignorata ---")


    # VALUTAZIONE SUL DATASET ANOMALIE 

    print("\n--- FASE 2: Lettura Dataset Anomalie ---")
    input_pattern_anom = os.path.expanduser(str(args.input[0]))
    files_anom = glob.glob(input_pattern_anom)
    print(f"Trovati {len(files_anom)} file anomalie.")

    # Liste salveranno solo i pixel utili
    val_labels_list = []
    val_msp_list = []
    val_logit_list = []
    val_entropy_list = []
    val_rba_list = []
    
    t_values = [0.1, 0.25, 0.5, 0.75, 0.8, 1.0, 1.1, 1.2, 1.5, 2.0, 5.0, 10.0]
    val_temp_list = {T: [] for T in t_values}

    for path in files_anom:
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().to(device)

        with torch.no_grad():
            if args.model_type == 'eomt':
                with autocast(device_type=device.type, dtype=torch.float16):
                    altezza_img, larghezza_img = images.shape[-2], images.shape[-1]
                    mask_logits_per_layer, class_logits_per_layer = model(images)

                    mask_logits = F.interpolate(mask_logits_per_layer[-1], size=(altezza_img, larghezza_img), mode="bilinear")
                    class_logits = class_logits_per_layer[-1]

                    mask_probs = mask_logits.sigmoid()
                    class_probs = F.softmax(class_logits, dim=-1)[..., :-1]
                    sem_seg_probs = torch.einsum("bqc, bqhw -> bchw", class_probs, mask_probs)

                    pixel_logits = torch.log(sem_seg_probs[0].float() + 1e-7)
                    rba_score = calculate_rba(pixel_logits)

            elif args.model_type == 'erfnet':
                result = model(images)
                pixel_logits = result.squeeze(0) 
                rba_score = None

            # Calcolo metriche standard 
            msp_score = calculate_msp(pixel_logits)
            logit_score = calculate_max_logit(pixel_logits)
            entropy_score = calculate_entropy(pixel_logits)
            
            # Calcolo Temperature 
            msp_t_scores_img = {T: calculate_msp(pixel_logits, temperature=T) for T in t_values}

        # Gestione Ground Truth
        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT: pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT: pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT: pathGT = pathGT.replace("jpg", "png")  

        if not os.path.exists(pathGT):
            continue

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)

        # Mappatura classi
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

        if 1 in np.unique(ood_gts):
            gt_flat = ood_gts.flatten()
            
            # Filtro
            mask_v = (gt_flat == 0) | (gt_flat == 1)
            
            if mask_v.any():
                val_labels_list.append(gt_flat[mask_v].astype(np.int8))
                val_msp_list.append(msp_score.flatten()[mask_v].astype(np.float32))
                val_logit_list.append(logit_score.flatten()[mask_v].astype(np.float32))
                val_entropy_list.append(entropy_score.flatten()[mask_v].astype(np.float32))
                
                if args.model_type == 'eomt':
                    val_rba_list.append(rba_score.flatten()[mask_v].astype(np.float32))
                
                for T in t_values:
                    val_temp_list[T].append(msp_t_scores_img[T].flatten()[mask_v].astype(np.float32))

        del images, pixel_logits
        torch.cuda.empty_cache()

    # CALCOLO METRICHE FINALI

    print("\n" + "="*50)
    print("--- CALCOLO METRICHE FINALI E TEMPERATURE ---")
    print("="*50)

    # I dati sono già filtrati
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

        print("\n--- TEST TEMPERATURE PER MSP (GRID SEARCH) ---")
        print(f"{'Temp':<8} | {'AUPRC (%)':<12} | {'FPR95 (%)':<12}")
        file.write("\nRISULTATI MSP CON TEMPERATURE:\n")

        for T in t_values:
            val_out_t = np.concatenate(val_temp_list[T])
            val_temp_list[T] = None
            
            prc_auc = average_precision_score(val_label, val_out_t)
            fpr = fpr_at_95_tpr(val_out_t, val_label)
            
            tipo = "(Standard)" if T == 1.0 else ""
            print(f"{T:<8.1f} | {prc_auc*100.0:<12.2f} | {fpr*100.0:<12.2f} {tipo}")
            file.write(f"T={T:.1f} -> AUPRC: {prc_auc*100.0:.2f} | FPR95: {fpr*100.0:.2f}\n")
            del val_out_t

    print("\nReport completo salvato in 'results.txt'")

if __name__ == '__main__':
    main()