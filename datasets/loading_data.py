import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
from .crowd_dataset import CrowdDataset
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split

# the function to return the dataloader 
def loading_data(args):
    # the augumentations
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
    ])
    # create  the dataset
    '''
    dataset = CrowdDataset(root_path=args.data_path, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    print(len(train_set))
    print(len(test_set))
    '''
    train_set = CrowdDataset(root_path=args.data_path, split='train', transform=transform)
    test_set = CrowdDataset(root_path=args.data_path, split='val', transform=transform)
    train_loader = DataLoader(train_set, batch_size=20, num_workers=4, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=20, num_workers=4, shuffle=False, drop_last=False)  
    return train_loader, test_loader

