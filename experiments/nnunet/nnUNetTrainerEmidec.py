"""
nnUNetTrainerEmidec.py

Custom nnU-Net v2 (2.8.1) trainers for the segmentation comparison
(experiments/SEGMENTATION_COMPARISON_PLAN.md). Copy this file into
nnunetv2/training/nnUNetTrainer/ before use (modal_nnunet.py and test_trainer.py do);
nnU-Net finds trainers by class name in that package.

  nnUNetTrainerEmidec            arms 1, 2, 4, 5. 1000 epochs x 250 iterations (nnU-Net default).
  nnUNetTrainerEmidecStrongAug   arm 3: same, with the stronger augmentation fixed below.

Environment variables (set per run):
  EMIDEC_SEED          integer seed (default 0). Seeds python, numpy and torch in the main
                       process, before the network is built, and the data-loader workers.
                       Not bit-exact: the multi-threaded augmenter hands out batches in
                       non-deterministic order and GPU kernels are not forced deterministic.
  EMIDEC_SYN_FRACTION  share of synthetic (insert) volumes in every training batch, e.g. 0.3333333
                       for arms 2, 4, 5 (exactly round(batch_size / 3) = 8 of 24 per batch,
                       the rest real, positions shuffled so insert samples are not always
                       the ones nnU-Net forces to contain foreground). 0 (default) for arms 1 and 3.
                       Synthetic volumes are the dataset cases whose id contains '__ins'.

Same for every arm: random in-plane crop-centre jitter (integer shift of up to +-8 px, image
filled with 0, segmentation with nnU-Net's ignore value -1, applied to the initial patch before
nnU-Net's own spatial transform) plus nnU-Net's standard augmentation.

Arm 3 stronger augmentation (declared before any run; the default values are in brackets):
  rotation          p 0.5 [0.2], range unchanged (the patch is isotropic: +-180 degrees)
  scaling           p 0.5 [0.2], (0.6, 1.6) [(0.7, 1.4)]
  elastic deform    p 0.3 [0], scale (0.1, 0.25) of the edge length, magnitude (0, 5) px
  gaussian noise    p 0.3 [0.1], variance (0, 0.15) [(0, 0.1)]
  gaussian blur     p 0.4 [0.2]
  brightness        p 0.4 [0.15], multiplier (0.6, 1.4) [(0.75, 1.25)]
  contrast          p 0.4 [0.15], (0.6, 1.4) [(0.75, 1.25)]
  low resolution    p 0.4 [0.25]
  gamma (inverted)  p 0.25 [0.1], (0.6, 1.7) [(0.7, 1.5)]
  gamma             p 0.5 [0.3], (0.6, 1.7) [(0.7, 1.5)]
  mirror            unchanged
"""

import os
import random

import numpy as np
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer import nnUNetTrainer as _upstream
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

SYN_MARK = "__ins"
JITTER_PX = 8


class RandomShiftTransform(BasicTransform):
    """Integer in-plane shift of image and segmentation, uniform in [-max_shift, max_shift] per axis."""

    def __init__(self, max_shift=JITTER_PX):
        super().__init__()
        self.max_shift = max_shift

    def get_parameters(self, **data_dict):
        m = self.max_shift
        return {"dy": int(torch.randint(-m, m + 1, (1,))), "dx": int(torch.randint(-m, m + 1, (1,)))}

    @staticmethod
    def _shift(x, dy, dx, fill):
        out = torch.full_like(x, fill)
        h, w = x.shape[-2:]
        sy_src, sy_dst = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
        sx_src, sx_dst = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
        out[..., sy_dst, sx_dst] = x[..., sy_src, sx_src]
        return out

    def _apply_to_image(self, img, **params):
        return self._shift(img, params["dy"], params["dx"], 0)

    def _apply_to_segmentation(self, seg, **params):
        return self._shift(seg, params["dy"], params["dx"], -1)


class MixedDataLoader(nnUNetDataLoader):
    """Every batch holds exactly round(batch_size * syn_fraction) synthetic and the rest real cases."""

    def __init__(self, *args, syn_fraction=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.syn_fraction = syn_fraction
        self.syn_keys = [k for k in self.indices if SYN_MARK in k]
        self.real_keys = [k for k in self.indices if SYN_MARK not in k]
        assert self.syn_keys and self.real_keys, "mixing needs both real and synthetic cases in the dataset"
        self.n_syn = int(round(self.batch_size * syn_fraction))

    def get_indices(self):
        keys = (list(np.random.choice(self.syn_keys, self.n_syn, replace=True))
                + list(np.random.choice(self.real_keys, self.batch_size - self.n_syn, replace=True)))
        np.random.shuffle(keys)
        return keys


class nnUNetTrainerEmidec(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        self.emidec_seed = int(os.environ.get("EMIDEC_SEED", 0))
        self.syn_fraction = float(os.environ.get("EMIDEC_SYN_FRACTION", 0.0))
        self._seed_all()
        super().__init__(plans, configuration, fold, dataset_json, device)

    def _seed_all(self):
        random.seed(self.emidec_seed)
        np.random.seed(self.emidec_seed)
        torch.manual_seed(self.emidec_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.emidec_seed)

    def initialize(self):
        self._seed_all()
        super().initialize()
        self.print_to_log_file(
            f"EMIDEC setup: trainer {type(self).__name__}, seed {self.emidec_seed}, synthetic fraction "
            f"{self.syn_fraction:.4f} (n per batch {int(round(self.batch_size * self.syn_fraction))} of {self.batch_size}), "
            f"epochs {self.num_epochs} x {self.num_iterations_per_epoch} iterations, jitter +-{JITTER_PX} px")

    @staticmethod
    def get_training_transforms(patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
                                do_dummy_2d_data_aug, use_mask_for_norm=None, is_cascaded=False,
                                foreground_labels=None, regions=None, ignore_label=None):
        base = nnUNetTrainer.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=use_mask_for_norm, is_cascaded=is_cascaded, foreground_labels=foreground_labels,
            regions=regions, ignore_label=ignore_label)
        return ComposeTransforms([RandomShiftTransform(JITTER_PX), base])

    def get_dataloaders(self):
        """Upstream get_dataloaders (nnU-Net 2.8.1) with the training loader swapped for MixedDataLoader
        when EMIDEC_SYN_FRACTION > 0, and the augmenter workers seeded."""
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        lm = self.label_manager
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm, is_cascaded=self.is_cascaded,
            foreground_labels=lm.foreground_labels, regions=lm.foreground_regions if lm.has_regions else None,
            ignore_label=lm.ignore_label)
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales, is_cascaded=self.is_cascaded, foreground_labels=lm.foreground_labels,
            regions=lm.foreground_regions if lm.has_regions else None, ignore_label=lm.ignore_label)
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        common = dict(oversample_foreground_percent=self.oversample_foreground_percent, sampling_probabilities=None,
                      pad_sides=None, probabilistic_oversampling=self.probabilistic_oversampling)
        if self.syn_fraction > 0:
            dl_tr = MixedDataLoader(dataset_tr, self.batch_size, initial_patch_size, patch_size, lm,
                                    transforms=tr_transforms, syn_fraction=self.syn_fraction, **common)
        else:
            assert not any(SYN_MARK in k for k in dataset_tr.identifiers), \
                "synthetic cases are in the dataset but EMIDEC_SYN_FRACTION is 0"
            dl_tr = nnUNetDataLoader(dataset_tr, self.batch_size, initial_patch_size, patch_size, lm,
                                     transforms=tr_transforms, **common)
        dl_val = nnUNetDataLoader(dataset_val, self.batch_size, patch_size, patch_size, lm,
                                  transforms=val_transforms, **common)
        n_proc = get_allowed_n_proc_DA()
        if n_proc == 0:
            return SingleThreadedAugmenter(dl_tr, None), SingleThreadedAugmenter(dl_val, None)
        pin = self.device.type == "cuda"
        seeds_tr = [self.emidec_seed * 1000 + i for i in range(n_proc)]
        n_val = max(1, n_proc // 2)
        seeds_val = [self.emidec_seed * 1000 + 500 + i for i in range(n_val)]
        mt_tr = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None, num_processes=n_proc,
                                             num_cached=max(6, n_proc // 2), seeds=seeds_tr, pin_memory=pin, wait_time=0.002)
        mt_val = NonDetMultiThreadedAugmenter(data_loader=dl_val, transform=None, num_processes=n_val,
                                              num_cached=max(3, n_proc // 4), seeds=seeds_val, pin_memory=pin, wait_time=0.002)
        _ = next(mt_tr)
        _ = next(mt_val)
        return mt_tr, mt_val


class nnUNetTrainerEmidecStrongAug(nnUNetTrainerEmidec):
    """Arm 3: the declared stronger augmentation (see the module docstring)."""

    @staticmethod
    def get_training_transforms(patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
                                do_dummy_2d_data_aug, use_mask_for_norm=None, is_cascaded=False,
                                foreground_labels=None, regions=None, ignore_label=None):
        assert not do_dummy_2d_data_aug and not is_cascaded and regions is None and use_mask_for_norm in (None, [False])
        u = _upstream
        t = [RandomShiftTransform(JITTER_PX)]
        t.append(u.SpatialTransform(
            patch_size, patch_center_dist_from_border=0, random_crop=False,
            p_elastic_deform=0.3, elastic_deform_scale=(0.1, 0.25), elastic_deform_magnitude=(0, 5),
            p_rotation=0.5, rotation=rotation_for_DA, p_scaling=0.5, scaling=(0.6, 1.6),
            p_synchronize_scaling_across_axes=1, bg_style_seg_sampling=False,
            border_mode_seg="constant", padding_value_seg=-1))
        t.append(u.RandomTransform(u.GaussianNoiseTransform(noise_variance=(0, 0.15), p_per_channel=1,
                                                            synchronize_channels=True), apply_probability=0.3))
        t.append(u.RandomTransform(u.GaussianBlurTransform(blur_sigma=(0.5, 1.), synchronize_channels=False,
                                                           synchronize_axes=False, p_per_channel=0.5, benchmark=True),
                                   apply_probability=0.4))
        t.append(u.RandomTransform(u.MultiplicativeBrightnessTransform(
            multiplier_range=u.BGContrast((0.6, 1.4)), synchronize_channels=False, p_per_channel=1), apply_probability=0.4))
        t.append(u.RandomTransform(u.ContrastTransform(
            contrast_range=u.BGContrast((0.6, 1.4)), preserve_range=True, synchronize_channels=False, p_per_channel=1),
            apply_probability=0.4))
        t.append(u.RandomTransform(u.SimulateLowResolutionTransform(
            scale=(0.5, 1), synchronize_channels=False, synchronize_axes=True, ignore_axes=None,
            allowed_channels=None, p_per_channel=0.5), apply_probability=0.4))
        t.append(u.RandomTransform(u.GammaTransform(
            gamma=u.BGContrast((0.6, 1.7)), p_invert_image=1, synchronize_channels=False, p_per_channel=1,
            p_retain_stats=1), apply_probability=0.25))
        t.append(u.RandomTransform(u.GammaTransform(
            gamma=u.BGContrast((0.6, 1.7)), p_invert_image=0, synchronize_channels=False, p_per_channel=1,
            p_retain_stats=1), apply_probability=0.5))
        if mirror_axes is not None and len(mirror_axes) > 0:
            t.append(u.MirrorTransform(allowed_axes=mirror_axes))
        t.append(u.RemoveLabelTansform(-1, 0))
        if deep_supervision_scales is not None:
            t.append(u.DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
        return ComposeTransforms(t)
