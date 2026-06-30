# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from typing import List, Optional
import torch.nn as nn
import torch.nn.functional as F

from training.mask_classification_loss import MaskClassificationLoss
from training.lightning_module import LightningModule

class MaskClassificationSemantic(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        ignore_idx: int = 255,
        lr: float = 1e-4,
        llrd: float = 0.8,
        llrd_l2_enabled: bool = True,
        lr_mult: float = 1.0,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,

        # Parameters for selecting the loss variant
        loss_variant: str = "ce",             # choose among "ce" | "logitnorm" "
        logitnorm_tau: float = 0.04,          # temperature for LogitNorm

        # Freezing strategy for fine-tuning
        freeze_strategy: str = "none",
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
        )

        self.save_hyperparameters(ignore=["_class_path"])

        self.ignore_idx = ignore_idx
        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh
        self.stuff_classes = range(num_classes)

        self.criterion = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            num_labels=num_classes,
            no_object_coefficient=no_object_coefficient,

            # Initialization of new parameters for loss variant selection
            loss_variant=loss_variant,
            logitnorm_tau=logitnorm_tau,
        )

        self.init_metrics_semantic(ignore_idx, self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1)

        # Application of the freezing strategy for fine-tuning
        self.freeze_strategy = freeze_strategy
        self._apply_freeze_strategy()

    # Freezing strategy for fine-tuning
    def _apply_freeze_strategy(self):

        """
        Apply the freezing strategy to optimize memory usage and training time during fine-tuning.
        
        This function allows blocking certain parts of the network based on `self.freeze_strategy`:

        - "none"        : No freezing. The entire network is trained.
        - "head_only"   : Freeze the entire pre-trained encoder. Only the final heads
                          and learnable queries are trainable.
        - "queries_only": Freeze all parameters except the learnable queries.
        """

        if self.freeze_strategy == "none":
            return

        head_keywords = ("class_head", "mask_head", "upscale", "q.")

        for name, p in self.network.named_parameters():
            if self.freeze_strategy == "head_only":
                trainable = any(k in name for k in head_keywords)
            elif self.freeze_strategy == "queries_only":
                trainable = name.startswith("q.")
            else:
                raise ValueError(
                    f"Unknown freeze_strategy: {self.freeze_strategy}. "
                    f"Valid values: 'none', 'head_only', 'queries_only'."
                )
            p.requires_grad = trainable

        n_train = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.network.parameters())
        pct = 100.0 * n_train / max(n_total, 1)
        print(
            f"[freeze_strategy={self.freeze_strategy}] "
            f"trainable params: {n_train:,} / {n_total:,} ({pct:.2f}%)"
        )

    def eval_step(
        self,
        batch,
        batch_idx=None,
        log_prefix=None,
    ):
        imgs, targets = batch

        img_sizes = [img.shape[-2:] for img in imgs]
        crops, origins = self.window_imgs_semantic(imgs)
        mask_logits_per_layer, class_logits_per_layer = self(crops)

        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
            crop_logits = self.to_per_pixel_logits_semantic(mask_logits, class_logits)
            logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)

            self.update_metrics_semantic(logits, targets, i)

            if batch_idx == 0:
                self.plot_semantic(
                    imgs[0], targets[0], logits[0], log_prefix, i, batch_idx
                )

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_semantic("val")

    def on_validation_end(self):
        self._on_eval_end_semantic("val")
