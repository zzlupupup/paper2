from monai import transforms

def get_transform():
    label_transform = transforms.Compose(
        [
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS", labels=(('L', 'R'), ('P', 'A'), ('I', 'S'))),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=2,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.2),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.2),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    unlabel_transform = transforms.Compose([
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS", labels=(('L', 'R'), ('P', 'A'), ('I', 'S'))),
            transforms.RandSpatialCropd(keys=['image', 'label'], roi_size=[96, 96, 96], random_size=False),
                          ])

    return label_transform, unlabel_transform