import torch
import matplotlib.pyplot as plt
from PIL import Image
import os


def preprocess_image(image, transform):
    image = Image.open(image).convert('RGB')
    return image, transform(image).unsqueeze(0)


def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        predictions = torch.nn.functional.softmax(output, dim=1)
        return predictions.cpu().numpy().flatten()


def visualize_predictions(original_image, predictions, class_names, relative_path):
    relative_path = relative_path.replace(' ', "_")
    save_dir = "../outputs/results"
    save_path = os.path.join(save_dir, relative_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    ax[0].imshow(original_image)
    ax[0].axis("off")
    ax[1].barh(class_names, predictions)
    ax[1].set_xlabel("Probability")
    ax[1].set_title("Class Predictions")
    ax[1].set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
