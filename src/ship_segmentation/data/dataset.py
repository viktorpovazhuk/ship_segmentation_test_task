import cv2
from torch.utils.data import Dataset


class ImagesDataset(Dataset):
    def __init__(self, images_dir, metadata_df, get_mask):
        self.images_dir = images_dir
        self.metadata_df = metadata_df
        self.image_ids = metadata_df["ImageId"].unique()
        self.get_mask = get_mask

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        img = cv2.imread(str(self.images_dir / img_id))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = self.get_mask(img, self.metadata_df, img_id)

        return img, mask
