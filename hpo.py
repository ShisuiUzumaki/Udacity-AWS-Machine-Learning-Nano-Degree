#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import ImageFile
import sys
import boto3
import io
import os
import logging
import argparse

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
        testing data loader and will get the test accuray/loss of the model
        Remember to include any debugging/profiling hooks that you might need
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        model.to(device)
        for data, target in test_loader:
            data=data.to(device)
            target=target.to(device)
            output=model(data)
            test_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


def train(model, train_loader, criterion, optimizer, device, epochs=2):
    '''
    TODO: Complete this function that can take a model and
        data loaders for training and will get train the model
        Remember to include any debugging/profiling hooks that you might need
    '''
    for epoch in range(epochs):
        model.to(device)
        model.train()
        for step, (data, labels) in enumerate(train_loader):
            data=data.to(device)
            labels=labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        step * len(data),
                        len(train_loader.dataset),
                        100.0 * step / len(train_loader),
                        loss.item(),
                        )
                )
    return model

def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), path)

def net():
    '''
    TODO: Complete this function that initializes your model
            Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 133))
    return model

def create_data_loaders(data_dir, ctransform, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    data = torchvision.datasets.ImageFolder(root=data_dir, transform=ctransform)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    return loader

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    device = torch.device(args.device)
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
#     train_dir = os.environ['SM_CHANNEL_TRAIN']
#     val_dir = os.environ['SM_CHANNEL_VAL']
#     test_dir = os.environ['SM_CHANNEL_TEST']
    train_dir = args.train_data_dir
    batch_size = args.batch_size
    epochs = args.epochs
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
#                                            transforms.Normalize([0.485, 0.456, 0.406], 
#                                                                 [0.229, 0.224, 0.225])
                                         ])
    train_loader = create_data_loaders(train_dir, train_transform, batch_size)
    model=train(model, train_loader, loss_criterion, optimizer, device, epochs)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test_dir = args.test_data_dir
    test_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
#                                            transforms.Normalize([0.485, 0.456, 0.406], 
#                                                                 [0.229, 0.224, 0.225])
                                         ])
    test_loader = create_data_loaders(test_dir, test_transforms ,batch_size)
    test(model, test_loader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    
    s3 = boto3.client('s3')
    buffer = io.BytesIO()
    torch.save(model, buffer)
    s3.put_object(Bucket="sagemaker-us-east-1-015775941522", Key="hpo_tuned_model/model_transfer.pt", Body=buffer.getvalue())

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="PyTorch dog images classification")
    parser.add_argument('--device', type=str, default=device, metavar='D',  help='availabell device type')
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.003, metavar="LR", help="learning rate (default: 1.0)"
    )
    parser.add_argument(
        "--train-data-dir",
        type=str,
        default=os.environ['SM_CHANNEL_TRAIN'],
        metavar="TDD",
        help="Training data directory",
    )
    parser.add_argument(
        "--test-data-dir",
        type=str,
        default=os.environ['SM_CHANNEL_TEST'],
        metavar="EDD",
        help="Test data directory",
    )
    parser.add_argument(
        "--val-data-dir",
        type=str,
        default=os.environ['SM_CHANNEL_VAL'],
        metavar="VDD",
        help="Test data directory",
    )
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    args=parser.parse_args()
    
    main(args)
