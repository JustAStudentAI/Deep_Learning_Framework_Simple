import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter


def setup_distributed(backend="nccl"):
    # Initialize the distributed environment and set the GPU using LOCAL_RANK
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    # Destroy the distributed process group
    dist.destroy_process_group()


class SimpleCNN(nn.Module):
    # A simple CNN for MNIST digit classification
    # Architecture: A convolutional block (conv, ReLU, max pooling) and two fully connected layers
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),  # 1 -> 32 channels, 3x3 kernel, stride 1
            nn.ReLU(),
            nn.MaxPool2d(2)         # 2x2 max pooling
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 13 * 13, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)


def save_checkpoint(model, optimizer, epoch, filename='checkpoint.pt'):
    # Save model and optimizer states
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),  # Extract underlying model from DDP
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    if dist.get_rank() == 0:
        print(f"Checkpoint saved at epoch {epoch}")


def load_checkpoint(model, optimizer, filename='checkpoint.pt'):
    # Load checkpoint if available and return the next epoch
    if os.path.exists(filename):
        checkpoint = torch.load(filename, map_location='cuda')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
        return start_epoch
    return 0


def main():
    # Set up distributed training
    setup_distributed()
    
    # Data loading: apply transform, download MNIST, and partition the data
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)
    
    # Model, optimizer, and loss function setup
    model = SimpleCNN().cuda()
    model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    start_epoch = load_checkpoint(model, optimizer)
    epochs = 5

    # Training loop
    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 100 == 0 and batch_idx > 0 and dist.get_rank() == 0:
                print(f"Rank {dist.get_rank()}, Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}], Loss: {running_loss/100:.3f}")
                running_loss = 0.0
        if dist.get_rank() == 0:
            save_checkpoint(model, optimizer, epoch)
    
    cleanup()


if __name__ == "__main__":
    main()
