import torch
import datasets
from torchvision import transforms


# A mock dataset, just for test.
# Do not use it to train!
def load_data(args, accelerator):
    dataset = datasets.load_dataset("lambdalabs/naruto-blip-captions")
    column_names = dataset["train"].column_names

    image_column = column_names[0]
    caption_column = column_names[1]

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["captions"] = [caption for caption in examples[caption_column]]
        return examples

    with accelerator.main_process_first():
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        captions = [example["captions"] for example in examples]
        return {"pixel_values": pixel_values, "captions": captions}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=0,
    )
    return train_dataset, train_dataloader