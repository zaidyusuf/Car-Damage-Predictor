import io
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

class_names = [
    'Front Breakage', 'Front Crushed', 'Front Normal', 
    'Rear Breakage', 'Rear Crushed', 'Rear Normal'
]

# 1. Added realistic cost mapping
REPAIR_ESTIMATES = {
    'Front Breakage': '₹15,000 - ₹40,000',
    'Front Crushed': '₹80,000 - ₹2,50,000+',
    'Front Normal': '₹0 (No repair needed)',
    'Rear Breakage': '₹12,000 - ₹35,000',
    'Rear Crushed': '₹75,000 - ₹2,00,000+',
    'Rear Normal': '₹0 (No repair needed)'
}

class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = models.resnet50(weights=None)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CarClassifierResNet()

# IMPORTANT: Ensure this path points exactly to wherever you placed your .pth file earlier!
model.load_state_dict(torch.load("saved_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(image_bytes: bytes) -> tuple:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        
        # 2. Look up both the string name and the cost
        predicted_name = class_names[predicted_class.item()]
        estimated_cost = REPAIR_ESTIMATES[predicted_name]
        
        # 3. Return BOTH variables so main.py does not crash
        return predicted_name, estimated_cost