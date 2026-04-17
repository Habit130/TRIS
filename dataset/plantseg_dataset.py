import json
import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

import CLIP.clip as clip


class PlantSegDataset(data.Dataset):
    def __init__(
        self,
        root="../plantseg",
        split="train",
        image_transforms=None,
        max_tokens=77,
        caption_index=3,
        eval_mode=False,
        size=320,
    ) -> None:
        self.root = root
        self.split = split
        self.image_transforms = image_transforms
        self.max_tokens = max_tokens
        self.caption_index = caption_index
        self.eval_mode = eval_mode
        self.size = size

        meta_path = os.path.join(root, "main.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            all_items = json.load(f)

        self.items = [item for item in all_items if item.get("split") == split]
        if not self.items:
            raise ValueError(f"No PlantSeg samples found for split '{split}' under {root}")

    def __len__(self):
        return len(self.items)

    def _tokenize_text(self, text):
        word_id = clip.tokenize(text, context_length=self.max_tokens, truncate=True).squeeze(0)
        word_mask = (word_id > 0).to(dtype=torch.int64)
        return word_id.unsqueeze(0), word_mask.unsqueeze(0)

    def __getitem__(self, index):
        item = self.items[index]
        image_path = os.path.join(self.root, item["image"])
        mask_path = os.path.join(self.root, item["mask"])

        img = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask_np = (np.asarray(mask) > 0).astype(np.uint8)
        mask = Image.fromarray(mask_np, mode="L")
        orig_h, orig_w = mask_np.shape

        if self.caption_index >= len(item["caption"]):
            raise IndexError(
                f"caption_index={self.caption_index} out of range for sample '{item['id']}'"
            )
        text = item["caption"][self.caption_index]
        word_ids, word_masks = self._tokenize_text(text)

        if self.image_transforms is not None:
            img, target = self.image_transforms(img, mask)
        else:
            target = torch.tensor(mask_np, dtype=torch.int64)

        samples = {
            "img": img,
            "word_ids": word_ids,
            "word_masks": word_masks,
        }
        targets = {
            "target": target.unsqueeze(0),
            "pseudo_gt": target.unsqueeze(0),
            "boxes": np.zeros(4, dtype=np.int64),
            "sentences": text,
            "img_path": item["id"],
            "img_path_full": item["image"],
            "mask_path_full": item["mask"],
            "orig_size": np.array([orig_h, orig_w]),
            "disease_label": item.get("disease_label", ""),
        }

        return samples, targets
