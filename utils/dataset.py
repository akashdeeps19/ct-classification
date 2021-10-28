import os
from PIL import Image
from torch.utils.data import Dataset,ConcatDataset
from torchvision import transforms
import torch

class BinaryCovid(Dataset):
    def __init__(self,image_root,gt_root,trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts =  [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images=sorted(self.images)
        self.gts=sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])


    def __getitem__(self,index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        image_tr = self.img_transform(image)
        gt_tr = self.gt_transform(gt)
        #image_concat = ConcatDataset([image_tr,image])
        #gt_concat = ConcatDataset([gt_tr,gt])
        return image_tr,gt_tr
    
    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts
        
    def rgb_loader(self,path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
            
    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt
    def __len__(self):
        return self.size


class EnsembleDataset(Dataset):
    def __init__(self,image1_root,image2_root,gt_root,trainsize):
        self.trainsize = trainsize
        self.images1 = [image1_root + f for f in os.listdir(image1_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images2 = [image2_root + f for f in os.listdir(image2_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts =  [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images1=sorted(self.images1)
        self.images2=sorted(self.images2)
        self.gts=sorted(self.gts)
        self.filter_files()
        self.size = len(self.images1)
        
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])


    def __getitem__(self,index):
        image1 = self.rgb_loader(self.images1[index])
        image2 = self.rgb_loader(self.images2[index])
        gt = self.binary_loader(self.gts[index])
        image_tr1 = self.img_transform(image1)
        image_tr2 = self.img_transform(image2)
        image_tr = torch.cat((image_tr1, image_tr2))
        gt_tr = self.gt_transform(gt)
        #image_concat = ConcatDataset([image_tr,image])
        #gt_concat = ConcatDataset([gt_tr,gt])
        return image_tr,gt_tr
    
    def filter_files(self):
        assert len(self.images1) == len(self.gts)
        images1 = []
        images2 = []
        gts = []
        for img1_path,img2_path, gt_path in zip(self.images1,self.images2, self.gts):
            img1 = Image.open(img1_path)
            img2 = Image.open(img2_path)
            gt = Image.open(gt_path)
            if img1.size == gt.size:
                images1.append(img1_path)
                images2.append(img2_path)
                gts.append(gt_path)
        self.images1 = images1
        self.images2 = images2
        self.gts = gts
        
    def rgb_loader(self,path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
            
    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt
    def __len__(self):
        return self.size
            
