import glob
import os

import torch.utils.data as data
from PIL import Image


class ADESegmentation(data.Dataset):
    def __init__(self, transforms=None, if_train: bool = True):
        super(ADESegmentation, self).__init__()
        root = "D:\Dataset\ADEChallengeData2016"
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        if if_train:
            this_dir = 'training'
        else:
            this_dir = 'validation'
        # 获取当前目录
        image_dir = os.path.join(root, 'images', this_dir)
        mask_dir = os.path.join(root, 'annotations', this_dir)

        self.images = sorted(glob.glob(image_dir + "/*.jpg"))
        self.masks = sorted(glob.glob(mask_dir + "/*.png"))
        assert (len(self.images) == len(self.masks))
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):  # 自定义每一个batch的输出格式，按自己的方式进行堆叠
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


dataset = ADESegmentation()
d1 = dataset[0]
print(d1)
