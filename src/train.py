import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import PlayingCardDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from model import CardClassifier
from tqdm import tqdm

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),

])
train_datas = "../datasets/cards/train"
test_datas = "../datasets/cards/test"
valid_datas = "../datasets/cards/valid"

train_dataset = PlayingCardDataset(train_datas, transform=transform)
test_dataset = PlayingCardDataset(test_datas, transform=transform)
valid_dataset = PlayingCardDataset(valid_datas, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 5

train_losses, valid_losses = [], []
model = CardClassifier(num_classes=53)
model.to(device)

# Loss Function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc="Training loop"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc="Validation loop"):
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
        valid_loss = running_loss / len(valid_loader.dataset)
        valid_losses.append(valid_loss)
    print(f"Epoch {epoch + 1}/{num_epochs} -Train loss :{train_loss:.4f} -Val loss :{valid_loss:.4f} ")

torch.save(model.state_dict(), "../models/card_classifier.pt")
