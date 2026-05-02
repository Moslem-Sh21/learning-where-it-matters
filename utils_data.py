import glob
import os
import json
import re
from PIL import Image
import numpy as np
import torch


def int_to_onehot(x, n):
    if not isinstance(x, list):
        x = [x]
    assert isinstance(x[0], int)
    x = torch.tensor(x).long()
    v = torch.zeros(n)
    v[x] = 1.0
    return v


random_select = lambda l: l[np.random.choice(len(l))]
top_select = lambda l: l[0]


NUMERIC_PATTERN = re.compile(r".*?(\d+)$")
# Matches: "42", "image_42", "person_0042" → extracts numeric suffix


def extract_numeric_id(filename):
    """Return integer id extracted from filename or None if no numeric suffix."""
    name = os.path.splitext(filename)[0]
    m = NUMERIC_PATTERN.match(name)
    return int(m.group(1)) if m else None


class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, transform, tokenizer, max_concept_length, select):
        # --------------------------------------------------
        # 1) Collect image files with numeric suffix
        # --------------------------------------------------
        image_paths = []
        for ext in [".jpg", ".jpeg", ".png"]:
            for file_path in glob.glob(os.path.join(image_folder, f"*{ext}")):
                filename = os.path.basename(file_path)
                numeric_id = extract_numeric_id(filename)
                if numeric_id is not None:
                    image_paths.append(file_path)

        if len(image_paths) == 0:
            raise ValueError(
                f"No images with numeric suffix found in {image_folder}.\n"
                "Expected formats include: 0001.jpg, image_0001.png, foo_42.jpeg"
            )

        # Sort by numeric id
        image_paths = sorted(image_paths, key=lambda p: extract_numeric_id(os.path.basename(p)))

        self.transform = transform
        self.tokenizer = tokenizer
        self.max_concept_length = max_concept_length

        # --------------------------------------------------
        # 2) Load concept dictionary
        # --------------------------------------------------
        concept_dict_path = os.path.join(image_folder, "concept_dict.json")
        if not os.path.exists(concept_dict_path):
            raise FileNotFoundError(f"concept_dict.json not found in {image_folder}")
        self.concept_dict = json.load(open(concept_dict_path, "r"))
        print(f"✓ Loaded concept dictionary with {len(self.concept_dict)} entries")

        # Select method for label entry
        if select == "top":
            self.select_method = top_select
        elif select == "random":
            self.select_method = random_select
        else:
            raise NotImplementedError(f"Unknown select method: {select}")

        # --------------------------------------------------
        # 3) Load labels.json
        # --------------------------------------------------
        labels_path = os.path.join(image_folder, "labels.json")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"labels.json not found in {image_folder}")
        labels = json.load(open(labels_path, "r"))
        print(f"✓ Loaded {len(labels)} label entries")

        # --------------------------------------------------
        # 4) Build aligned samples
        # --------------------------------------------------
        self.samples = []
        skipped = 0

        for img_path in image_paths:
            basename = os.path.basename(img_path)
            idx = extract_numeric_id(basename)

            if idx is None or idx < 0 or idx >= len(labels):
                skipped += 1
                continue

            self.samples.append((img_path, labels[idx]))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid (image, label) pairs in {image_folder}. "
                "Check filename numbering and labels.json length."
            )

        print(f"✓ Using {len(self.samples)} aligned (image, label) pairs")
        if skipped > 0:
            print(f"⚠️ Skipped {skipped} images with out-of-range or invalid IDs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label_entry = self.samples[index]

        # label_entry: [prompt, [concept_names]]
        input_prompt, target_concept_names = self.select_method(label_entry)

        # Tokenize prompt
        input_prompt_tokens = self.tokenizer([input_prompt])[0]

        # Map concept names -> indices
        mapped_indices = [self.concept_dict[c] for c in target_concept_names if c in self.concept_dict]
        if not mapped_indices:
            raise ValueError(
                f"Unknown target concepts {target_concept_names}. "
                f"Available concepts: {list(self.concept_dict.keys())}"
            )

        target_concept = int_to_onehot(mapped_indices, self.max_concept_length)

        # Load and transform image
        x = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            x = self.transform(x)

        return x, input_prompt_tokens, target_concept


def get_dataloader(
    image_folder,
    batch_size,
    transform,
    tokenizer,
    collate_fn=None,
    num_workers=4,
    shuffle=False,
    max_concept_length=100,
    select="random",
):
    dataset = TrainingDataset(
        image_folder,
        transform=transform,
        tokenizer=tokenizer,
        select=select,
        max_concept_length=max_concept_length,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
