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

    # Usare IsoMax+
    parser.add_argument('--use_isomax', action='store_true', help="Usa la testa IsoMax+ invece di quella standard")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device in uso: {device} \n")

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

    print(f"INIZIO VALUTAZIONE CON MODELLO: {args.model_type.upper()} \n")

    # Scelgo il modello da caricare
    if args.model_type == 'eomt':

        encoder = ViT(img_size=img_size, patch_size=16, backbone_name="vit_base_patch14_reg4_dinov2") 
        model = EoMT(encoder=encoder, num_classes=19, num_q=100, num_blocks=3, masked_attn_enabled=False, use_isomax_plus_head=True if args.use_isomax else False) 

        weightspath = os.path.join(args.loadDir, args.loadWeights)
        print(f"\nCaricamento pesi EoMT da: {weightspath}")

        # Carichiamo lo state_dict nativo dal file .bin
        state_dict_raw = torch.load(weightspath, map_location='cpu', weights_only=False)

        # Estraiamo i pesi se è un file Lightning
        if 'state_dict' in state_dict_raw:
            state_dict = state_dict_raw['state_dict']
            print("Rilevato checkpoint PyTorch Lightning (.ckpt). Estraggo i pesi...")
        else:
            state_dict = state_dict_raw
            print("Rilevato checkpoint PyTorch standard (.bin).")

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

        print("\nLettura Cityscapes (Immagini Normali ID)")
        input_pattern_city = os.path.expanduser(str(args.input_cityscapes))
        files_city = glob.glob(input_pattern_city)
        print(f"Trovate {len(files_city)} immagini Cityscapes.")

        # Inizializziamo la classe (20 classi totali, ignoreIndex = 19)
        iouEvalVal = iouEval.iouEval(nClasses=20, ignoreIndex=19)

        for path in files_city:

            #print(f"Cityscapes ID: {os.path.basename(path)}")
            images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().to(device)

            # Carichiamo solo i labelTrainIds (che hanno classi da 0 a 18, più 255 da ignorare), 
            pathGT = path.replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png").replace("leftImg8bit", "gtFine")
            
            # Non usiamo labelIds.png, contiene 34 classi diverse (con indici sballati)
            # invece delle 19 standard di Cityscapes (che si trovano in labelTrainIds.png)
            if not os.path.exists(pathGT):
                print(f"Salto {os.path.basename(path)}: labelTrainIds non trovato.")
                continue

            mask = target_transform(Image.open(pathGT))
            
            # Forziamo la GT a essere un 2D
            gt_arr = np.array(mask)
            if gt_arr.ndim == 3:
                gt_arr = gt_arr[:, :, 0]

            # Creiamo il tensore con shape [B, 1, H, W] richiesto dalla classe
            gt_tensor = torch.from_numpy(gt_arr).long().to(device).unsqueeze(0).unsqueeze(0)

            # Rimappiamo il 255 a 19 per farlo gestire dal 'ignoreIndex=19'
            gt_tensor[gt_tensor == 255] = 19

            with torch.no_grad():
                with autocast(dtype=torch.float16, device_type="cuda"):

                    # prende dim immagini, le fa passare a eomt, torna dove sono gli oggetti e che classi sono
                    altezza_img, larghezza_img = images.shape[-2], images.shape[-1]
                    mask_logits_per_layer, class_logits_per_layer = model(images)

                    # prende risultati dell'ultimo strato di rete, stira le maschere riportandole alla risoluzione originale
                    mask_logits = F.interpolate(mask_logits_per_layer[-1], size=(altezza_img, larghezza_img), mode="bilinear")

                    if args.use_isomax:
                        # Moltiplichiamo per l'entropic_scale (10.0) come in fase di training per ottenere le stesse probabilità 
                        class_logits = class_logits_per_layer[-1] * 10.0
                    else:
                        class_logits = class_logits_per_layer[-1]

                    # le maschere vengono passate in una sigmoid (diventano probabilità tra 0 e 1)
                    # le classi passano in una softmax (diventano percentuali)
                    # [..., :-1] butta via l'ultimissima classe (lo "sfondo/void")
                    mask_probs = mask_logits.sigmoid()
                    class_probs = F.softmax(class_logits, dim=-1)[..., :-1]

                    # moltiplica le probabilità delle classi con le probabilità delle maschere, creando la mappa di segmentazione finale
                    sem_seg_probs = torch.einsum("bqc, bqhw -> bchw", class_probs, mask_probs)
                    
                    # Troviamo la classe vincente prevista dalla rete per calcolare la mIoU
                    preds = torch.argmax(sem_seg_probs, dim=1, keepdim=True)

                # Passiamo i tensori [1, 1, H, W] alla classe
                iouEvalVal.addBatch(preds, gt_tensor)

                del images, sem_seg_probs, preds, gt_tensor
                torch.cuda.empty_cache()

        # Recuperiamo e stampiamo i risultati dalla tua classe
        miou_val, iou_classes = iouEvalVal.getIoU()
        
        print("\n" + "="*40)
        print(f"Cityscapes mIoU: {miou_val.item() * 100.0:.2f}%")
        print("="*40)

        # Stampa delle singole classi per debug
        """
        class_names = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", 
                        "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
        
        iou_classes_np = iou_classes.cpu().numpy() * 100
        print("IoU per singola classe:")
        for i, name in enumerate(class_names):
            if i < len(iou_classes_np):
                print(f"{name:15s}: {iou_classes_np[i]:.2f}%")
        """

    # VALUTAZIONE SUL DATASET ANOMALIE 

    print("\nLettura Dataset Anomalie")
    input_pattern_anom = os.path.expanduser(str(args.input[0]))
    files_anom = glob.glob(input_pattern_anom)
    print(f"Trovati {len(files_anom)} file anomalie.")

    # Liste salveranno solo i pixel utili
    val_labels_list = []
    val_msp_list = []
    val_logit_list = []
    val_entropy_list = []
    val_rba_list = []
    
    # t_values = [0.1, 0.25, 0.5, 0.75, 0.8, 1.0, 1.1, 1.2, 1.5, 2.0, 5.0, 10.0]
    t_values = [0.1, 0.25, 0.5, 0.75, 0.8, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 4.0,
                5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0]
    val_temp_list = {T: [] for T in t_values}

    for path in files_anom:
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().to(device)

        with torch.no_grad():
            if args.model_type == 'eomt':
                with autocast(device_type=device.type, dtype=torch.float16):
                    altezza_img, larghezza_img = images.shape[-2], images.shape[-1]
                    mask_logits_per_layer, class_logits_per_layer = model(images)

                    mask_logits = F.interpolate(mask_logits_per_layer[-1], size=(altezza_img, larghezza_img), mode="bilinear")
                    
                    if args.use_isomax:
                        # Caso Isomax
                        class_logits = class_logits_per_layer[-1]
                        mask_probs = mask_logits.sigmoid()
                        
                        pixel_logits_tensor = torch.einsum("bqc, bqhw -> bchw", class_logits[..., :-1], mask_probs)
                        pixel_logits = pixel_logits_tensor[0].float()
                    else:
                        # Caso Standard (LogitNorm o Classico - Mantiene intatta la logica precedente)
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
    print("CALCOLO METRICHE FINALI")
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

        print("\nTEST TEMPERATURE PER MSP (GRID SEARCH)")
        print(f"{'Temp':<8} | {'AUPRC (%)':<12} | {'FPR95 (%)':<12}")
        file.write("\nRISULTATI MSP CON TEMPERATURE:\n")

        
        # QUA ABBIAMO LA STAMPA STATICA
        for T in t_values:
            val_out_t = np.concatenate(val_temp_list[T])
            val_temp_list[T] = None
            
            prc_auc = average_precision_score(val_label, val_out_t)
            fpr = fpr_at_95_tpr(val_out_t, val_label)
            
            tipo = "(Standard)" if T == 1.0 else ""
            print(f"{T:<8.1f} | {prc_auc*100.0:<12.2f} | {fpr*100.0:<12.2f} {tipo}")
            file.write(f"T={T:.1f} -> AUPRC: {prc_auc*100.0:.2f} | FPR95: {fpr*100.0:.2f}\n")
            del val_out_t
        

        '''
        # STAMPA DINAMICA CON STOP AUTOMATICO
        # ------------------------------------
        # Scorriamo le temperature in ordine crescente. Calcoliamo e stampiamo
        # le metriche finche' la curva migliora. Lo stop scatta al PRIMO punto in
        # cui ENTRAMBE le metriche peggiorano contemporaneamente rispetto al punto
        # precedente, cioe' AUPRC scende E FPR95 sale (per FPR95 "peggio" = piu' alto).
        # Dopo lo stop stampiamo ancora TAIL_POINTS punti per mostrare la discesa.
        TAIL_POINTS = 5         # punti di coda da stampare dopo lo stop
        EPS = 1e-9               # tolleranza per evitare stop su rumore numerico

        sorted_temps = sorted(t_values)
        prev_auprc = None
        prev_fpr = None
        stop_triggered = False   # True dopo il primo calo congiunto
        tail_remaining = 0       # quanti punti di coda restano da stampare

        for T in sorted_temps:
            val_out_t = np.concatenate(val_temp_list[T])
            val_temp_list[T] = None

            prc_auc = average_precision_score(val_label, val_out_t) * 100.0
            fpr = fpr_at_95_tpr(val_out_t, val_label) * 100.0
            del val_out_t

            # Verifica del calo congiunto (solo se abbiamo un punto precedente
            # e non abbiamo gia' fatto scattare lo stop)
            if not stop_triggered and prev_auprc is not None:
                auprc_peggiora = prc_auc < prev_auprc - EPS   # AUPRC scesa
                fpr_peggiora = fpr > prev_fpr + EPS           # FPR95 salita
                if auprc_peggiora and fpr_peggiora:
                    stop_triggered = True
                    tail_remaining = TAIL_POINTS

            # Stampa della riga corrente
            note = " (Standard)" if abs(T - 1.0) < EPS else ""
            if stop_triggered:
                note += " <-- calo (AUPRC giu, FPR95 su)" if tail_remaining == TAIL_POINTS else " (coda)"

            print(f"{T:<8.2f} | {prc_auc:<12.2f} | {fpr:<12.2f}{note}")
            file.write(f"T={T:.2f} -> AUPRC: {prc_auc:.2f} | FPR95: {fpr:.2f}{note}\n")

            # Aggiorniamo i valori precedenti per il prossimo confronto
            prev_auprc = prc_auc
            prev_fpr = fpr

            # Gestione dello stop + coda: una volta scattato lo stop, scaliamo
            # i punti di coda e ci fermiamo quando sono esauriti.
            if stop_triggered:
                tail_remaining -= 1
                if tail_remaining <= 0:
                    break

        print("-" * 40)
        if stop_triggered:
            print(f"Stop automatico: AUPRC e FPR95 hanno iniziato a peggiorare insieme "
                  f"(+ {TAIL_POINTS} punti di coda).")
        else:
            print("Nessun calo congiunto rilevato: la curva non ha ancora invertito "
                  "entro la griglia testata (prova temperature piu' alte).")
        '''

    print("\nReport completo salvato in 'results.txt'")

if __name__ == '__main__':
    main()