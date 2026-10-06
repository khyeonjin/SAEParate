import os
from glob import glob

import timm
import torch
from torchvision import transforms

torch.hub.set_dir("cache")
import sys

import fire
from PIL import Image
from tqdm import tqdm

sys.path.append("")
from UnlearnCanvas_resources.const import class_available, theme_available


def main(
    input_dir,
    output_dir,
    class_ckpt,
    style_ckpt,
    cls=None,
    seed=[42],
    dry_run=False,
    limit_classes=-1,
    batch_size=32,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dir = os.path.join(input_dir, cls) if cls is not None else input_dir

    os.makedirs(output_dir, exist_ok=True)

    # Initialize models
    class_model = timm.create_model(
        "vit_large_patch16_224.augreg_in21k", pretrained=True
    ).to(device)
    class_model.head = torch.nn.Linear(1024, len(class_available)).to(device)
    class_model.load_state_dict(
        torch.load(class_ckpt, map_location=device)["model_state_dict"]
    )
    class_model.eval()

    style_model = timm.create_model(
        "vit_large_patch16_224.augreg_in21k", pretrained=True
    ).to(device)
    style_model.head = torch.nn.Linear(1024, len(theme_available)).to(device)
    style_model.load_state_dict(
        torch.load(style_ckpt, map_location=device)["model_state_dict"]
    )
    style_model.eval()

    class_results = {
        "test_theme": cls if cls is not None else "sd",
        "input_dir": input_dir,
        "loss": {class_: 0.0 for class_ in class_available},
        "acc": {class_: 0.0 for class_ in class_available},
        "pred_loss": {class_: 0.0 for class_ in class_available},
        "misclassified": {
            class_: {other_class: 0 for other_class in class_available}
            for class_ in class_available
        },
    }

    style_results = {
        "test_theme": cls if cls is not None else "sd",
        "input_dir": input_dir,
        "loss": {theme: 0.0 for theme in theme_available},
        "acc": {theme: 0.0 for theme in theme_available},
        "pred_loss": {theme: 0.0 for theme in theme_available},
        "misclassified": {
            theme: {other_theme: 0 for other_theme in theme_available}
            for theme in theme_available
        },
    }

    image_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, image_paths, labels):
            self.image_paths = image_paths
            self.labels = labels

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path)
            image = image_transform(image)
            return image, self.labels[idx]

    classes_to_use = (
        class_available[:limit_classes] if limit_classes > 0 else class_available
    )
    class_label_map = {class_: idx for idx, class_ in enumerate(classes_to_use)}
    style_label_map = {theme: idx for idx, theme in enumerate(theme_available)}

    class_image_paths = []
    class_labels = []
    style_image_paths = []
    style_labels = []

    for s in seed:
        for theme in theme_available:
            if theme == "Seed_Images":
                continue
            for object_class in classes_to_use:
                for f in glob(os.path.join(input_dir, f"{theme}_{object_class}_seed{s}_*.jpg")):
                    class_image_paths.append(f)
                    class_labels.append(class_label_map[object_class])
                    style_image_paths.append(f)
                    style_labels.append(style_label_map[theme])

    class_dataset = ImageDataset(class_image_paths, class_labels)
    class_dataloader = torch.utils.data.DataLoader(
        class_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    style_dataset = ImageDataset(style_image_paths, style_labels)
    style_dataloader = torch.utils.data.DataLoader(
        style_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    for batch_images, batch_class_labels in tqdm(class_dataloader, desc="Class eval"):
        batch_images = batch_images.to(device)
        batch_class_labels = batch_class_labels.to(device)

        with torch.no_grad():
            class_res = class_model(batch_images)
            class_loss = torch.nn.functional.cross_entropy(
                class_res, batch_class_labels, reduction="none"
            )
            class_softmax = torch.nn.functional.softmax(class_res, dim=1)
            class_pred_labels = torch.argmax(class_res, dim=1)
            class_pred_success = class_pred_labels == batch_class_labels

            for i in range(len(batch_class_labels)):
                object_class = class_available[batch_class_labels[i].item()]
                class_results["loss"][object_class] += class_loss[i].item()
                class_results["pred_loss"][object_class] += class_softmax[i][
                    batch_class_labels[i]
                ].item()
                class_results["acc"][object_class] += class_pred_success[i].item()
                misclassified_as = class_available[class_pred_labels[i].item()]
                class_results["misclassified"][object_class][misclassified_as] += 1

    for object_class in class_available:
        total = sum(class_results["misclassified"][object_class].values())
        if total > 0:
            class_results["acc"][object_class] /= total

    for batch_images, batch_style_labels in tqdm(style_dataloader, desc="Style eval"):
        batch_images = batch_images.to(device)
        batch_style_labels = batch_style_labels.to(device)

        with torch.no_grad():
            style_res = style_model(batch_images)
            style_loss = torch.nn.functional.cross_entropy(
                style_res, batch_style_labels, reduction="none"
            )
            style_softmax = torch.nn.functional.softmax(style_res, dim=1)
            style_pred_labels = torch.argmax(style_res, dim=1)
            style_pred_success = style_pred_labels == batch_style_labels

            for i in range(len(batch_style_labels)):
                theme = theme_available[batch_style_labels[i].item()]
                style_results["loss"][theme] += style_loss[i].item()
                style_results["pred_loss"][theme] += style_softmax[i][
                    batch_style_labels[i]
                ].item()
                style_results["acc"][theme] += style_pred_success[i].item()
                misclassified_as = theme_available[style_pred_labels[i].item()]
                style_results["misclassified"][theme][misclassified_as] += 1

    for theme in theme_available:
        total = sum(style_results["misclassified"][theme].values())
        if total > 0:
            style_results["acc"][theme] /= total

    if not dry_run:
        torch.save(class_results, os.path.join(output_dir, f"{cls}_cls.pth"))
        torch.save(style_results, os.path.join(output_dir, f"{cls}.pth"))


if __name__ == "__main__":
    fire.Fire(main)
