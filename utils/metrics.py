import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os

def binary_loader( path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

def load_images(root,size):
    images =  [root + f for f in os.listdir(root) if f.endswith('.png')]
    print(len(images))
    images=sorted(images)
    transform = transforms.Compose([transforms.Resize((size, size)),transforms.ToTensor()])
    ims = []
    for image in images:
        ims.append(transform(binary_loader(image)))
    return ims

def dice(outputs: torch.Tensor, labels: torch.Tensor):
    dice = (2*(outputs*labels).sum())/((outputs+labels).sum()+1e-8)
    return dice

def get_dice(preds, actual):
    dice_score = 0
    for (pred, output) in zip(preds, actual):
        dice_score += dice(pred, output)
    return dice_score/len(preds)

def iou(prediction: torch.Tensor, truth: torch.Tensor):
    true_positives, false_positives, true_negatives, false_negatives = confusion(prediction,truth)
    iou = 1 - (true_positives/(true_positives+false_positives+false_negatives))
    
    return iou

def get_iou(preds, actual):
    IoU = 0
    for (pred, output) in zip(preds, actual):
        IoU += iou(pred, output)
    return IoU


def confusion(prediction, truth):
    confusion_vector = prediction / truth
    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives

def Fscore(prediction, truth):
    true_positives, false_positives, true_negatives, false_negatives = confusion(prediction,truth)
    # print(true_positives, false_positives, true_negatives, false_negatives)
    precision_pos = true_positives/(true_positives+false_positives+1e-8)
    recall_pos = true_positives/(true_positives+false_negatives+1e-8)
    precision_neg = true_negatives/(true_negatives+false_negatives+1e-8)
    recall_neg = true_negatives/(true_negatives+false_positives+1e-8)
    F1_pos = true_positives/(true_positives+false_negatives)
    F1_neg = true_negatives/(true_negatives+false_positives)
    return F1_pos,F1_neg

def get_fscore(preds, actual):
    f1_pos = 0
    f1_neg = 0
    for (pred, output) in zip(preds, actual):
        F1_pos,F1_neg=Fscore(pred,output)
        f1_pos+=F1_pos
        f1_neg+=F1_neg
    return f1_pos/len(preds), f1_neg/len(preds)

def mae(outputs: torch.Tensor, labels: torch.Tensor):
    return (outputs - labels).abs().mean()

def get_mae(preds, actual):
    MAE = 0
    for (pred, output) in zip(preds, actual):
        MAE += mae(pred, output)
    return MAE/len(preds)

def get_metrics(pred_path, actual_path, tsize):
    preds = load_images(pred_path,tsize)
    actual = load_images(actual_path,tsize)
    metrics = {}
    metrics['dice'] = get_dice(preds, actual)
    metrics['sensitivity'], metrics['specificity'] = get_fscore(preds, actual)
    metrics['iou'] = get_iou(preds, actual)
    metrics['mae'] = get_mae(preds, actual)
    return metrics