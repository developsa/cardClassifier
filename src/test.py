import torch
import torchvision.transforms as transforms
import numpy as np
from glob import glob
from torchvision.datasets import ImageFolder
from model import CardClassifier
from utils import preprocess_image, predict, visualize_predictions
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
model = CardClassifier(num_classes=53).to(device)
model.load_state_dict(torch.load('../models/card_classifier.pt', map_location=device))
model.eval()

dataset = ImageFolder('../datasets/cards/train', transform=transforms.ToTensor())
class_names = dataset.classes

# Examples
test_images = glob("../datasets/cards/test/*/*")
test_samples = np.random.choice(test_images, 10)

for example in test_samples:
    relative_path = os.path.relpath(example, start="../datasets/cards/test")
    relative_path = relative_path.replace("\\", "/")
    original_image, image_tensor = preprocess_image(example, transform)
    prediction = predict(model, image_tensor, device)
    visualize_predictions(original_image, prediction, class_names,relative_path)
