import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import datasets

### 1. Data Pipeline
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean, std=std)
transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), normalize])

trainset = datasets.HymenopteraDataset(root=args.data_root + '/train', transform=transform)
trainloader = DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True)
