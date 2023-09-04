import numpy as np
import cv2

from torch.utils.data import Dataset


class SiameseNetworkDataset(Dataset):
        """
        Creates a Dataset of image pairs of similar and different class
        """
        def __init__(self,csv_path, data_dir_path,transform=None):
            self.data_dir_path = data_dir_path
            self.transform = transform
            self.df = pd.read_csv(csv_path)

        def __len__(self):
            return len(self.df)

        def __getitem__(self,idx):
            img1_name, img2_name, label = self.df.iloc[idx]

            img1_path = os.path.join(self.data_dir_path,img1_name)
            img2_path = os.path.join(self.data_dir_path,img2_name)

            img1 = cv2.imread(img1_path)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img1 = Image.fromarray(img1)

            img2 = cv2.imread(img2_path)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            img2 = Image.fromarray(img2)

            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            return img1,img2, label

        
class TripletNetworkDataset(Dataset):
        """
        Creates a Dataset of image triplets having Anchor, Positive and Negative images 
        """
        def __init__(self,csv_path, data_dir_path,transform=None):
            self.data_dir_path = data_dir_path
            self.transform = transform
            self.df = pd.read_csv(csv_path)

        def __len__(self):
            return len(self.df)

        def __getitem__(self,idx):
            anchor_name, pos_name, neg_name = self.df.iloc[idx]

            anchor_path = os.path.join(self.data_dir_path,anchor_name)
            pos_path = os.path.join(self.data_dir_path,pos_name)
            neg_path = os.path.join(self.data_dir_path,neg_name)

            anchor = cv2.imread(anchor_path)
            anchor = cv2.cvtColor(anchor, cv2.COLOR_BGR2RGB)
            anchor = Image.fromarray(anchor)

            pos = cv2.imread(pos_path)
            pos = cv2.cvtColor(pos, cv2.COLOR_BGR2RGB)
            pos = Image.fromarray(pos)

            neg = cv2.imread(neg_path)
            neg = cv2.cvtColor(neg, cv2.COLOR_BGR2RGB)
            neg = Image.fromarray(neg)

            if self.transform:
                anchor = self.transform(anchor)
                pos = self.transform(pos)
                neg = self.transform(neg)
            return anchor,pos,neg
