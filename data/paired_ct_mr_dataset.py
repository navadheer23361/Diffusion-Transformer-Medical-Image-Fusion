"""Paired CT/MR PNG dataset utilities for DiffTransFuse.

Expected layout:
    root/
      patient_id/
        ct/001.png
        mr/001.png

The returned keys keep DM-FNet compatibility:
    ir  -> CT tensor, shape (1, H, W)
    vis -> MR tensor, shape (1, H, W)
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PairedCTMRDataset(Dataset):
    """Load paired CT and MR slices using the repository dataset layout."""

    def __init__(self, opt, split="train", patient_ids=None):
        """Create a patient-split CT/MR PNG dataset from the configured root."""
        self.root = Path(opt.get("data_root", "DATASET"))
        self.ct_folder = opt.get("ct_folder", "ct")
        self.mr_folder = opt.get("mr_folder", "mr")
        resolution = opt.get("resolution", [224, 256])
        if isinstance(resolution, int):
            self.image_h = resolution
            self.image_w = resolution
        else:
            self.image_h = int(resolution[0])
            self.image_w = int(resolution[1])

        self.split = split.lower()
        patients = sorted(p for p in self.root.iterdir() if p.is_dir())
        if patient_ids is not None:
            allowed = set(patient_ids)
            patients = [p for p in patients if p.name in allowed]
        else:
            patients = self._split_patients(patients, opt)

        self.samples = []
        for patient_dir in patients:
            ct_dir = patient_dir / self.ct_folder
            mr_dir = patient_dir / self.mr_folder
            if not ct_dir.is_dir() or not mr_dir.is_dir():
                continue
            ct_slices = {p.name: p for p in ct_dir.glob("*.png")}
            mr_slices = {p.name: p for p in mr_dir.glob("*.png")}
            for slice_name in sorted(ct_slices.keys() & mr_slices.keys()):
                self.samples.append(
                    {
                        "patient_id": patient_dir.name,
                        "slice_id": Path(slice_name).stem,
                        "ct_path": ct_slices[slice_name],
                        "mr_path": mr_slices[slice_name],
                    }
                )

        if not self.samples:
            raise RuntimeError(f"No paired CT/MR PNG slices found under {self.root}")

    def _split_patients(self, patients, opt):
        """Split patients deterministically into train, val, and test sets."""
        if self.split in ("all", "full"):
            return patients

        split_ratio = opt.get("split_ratio", [0.7, 0.15, 0.15])
        if len(split_ratio) != 3:
            raise ValueError("split_ratio must be [train, val, test]")

        seed = int(opt.get("split_seed", 42))
        patients = list(patients)
        rng = random.Random(seed)
        rng.shuffle(patients)

        total = len(patients)
        n_train = int(total * float(split_ratio[0]))
        n_val = int(total * float(split_ratio[1]))

        train_patients = patients[:n_train]
        val_patients = patients[n_train:n_train + n_val]
        test_patients = patients[n_train + n_val:]

        if self.split == "train":
            return sorted(train_patients)
        if self.split in ("val", "valid", "validation"):
            return sorted(val_patients)
        if self.split == "test":
            return sorted(test_patients)
        raise ValueError(f"Unknown split '{self.split}'. Use train, val, test, or all.")

    def __len__(self):
        return len(self.samples)

    def _load_grayscale(self, path):
        """Load one grayscale image and resize it to the configured resolution."""
        image = Image.open(path).convert("L")
        if image.size != (self.image_w, self.image_h):
            image = image.resize((self.image_w, self.image_h), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)

    def __getitem__(self, index):
        sample = self.samples[index]
        ct = self._load_grayscale(sample["ct_path"])
        mr = self._load_grayscale(sample["mr_path"])
        return {
            "ir": ct,
            "vis": mr,
            "ct": ct,
            "mr": mr,
            "patient_id": sample["patient_id"],
            "slice_id": sample["slice_id"],
            "ct_path": str(sample["ct_path"]),
            "mr_path": str(sample["mr_path"]),
        }
