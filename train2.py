# Task 1.2 - Improved version
# Fine-tunes a pretrained ResNet18 model for pneumonia detection

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# -------------------------
# Set random seed so results are reproducible
# -------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Data preprocessing + augmentation
# -------------------------

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),   # simple augmentation
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1),
    transforms.ToTensor(),

    # ImageNet normalization (needed for pretrained networks)
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------------
# Dataset paths
# -------------------------

train_dir = "/content/chest_xray/train"
val_dir   = "/content/chest_xray/val"
test_dir  = "/content/chest_xray/test"

trainset = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
valset   = torchvision.datasets.ImageFolder(val_dir, transform=val_transform)
testset  = torchvision.datasets.ImageFolder(test_dir, transform=val_transform)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True
)

valloader = torch.utils.data.DataLoader(
    valset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
)


# -------------------------
# Training parameters
# -------------------------

EPOCHS = 15
BATCH_SIZE = 16
LR = 1e-4

print("\nTraining setup")
print("Model: ResNet18 (ImageNet pretrained)")
print("Fine-tuning: layer4 + final FC layer")
print("Optimizer: AdamW")
print("LR:", LR)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Loss: CrossEntropy + label smoothing")
print("Scheduler: CosineAnnealingLR")
print("Device:", device)
print("Classes:", trainset.classes)
print()


# -------------------------
# Handle class imbalance
# -------------------------

class_counts = np.bincount(trainset.targets)
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
class_weights = class_weights.to(device)


# -------------------------
# Load pretrained ResNet18
# -------------------------

from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Freeze most layers (use pretrained features)
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last residual block for fine-tuning
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace the classifier for 2 classes
model.fc = nn.Linear(model.fc.in_features, len(trainset.classes))

model = model.to(device)


# -------------------------
# Loss function
# -------------------------

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1
)


# -------------------------
# Optimizer + LR scheduler
# -------------------------

optimizer = optim.AdamW(
    list(model.layer4.parameters()) + list(model.fc.parameters()),
    lr=LR,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS
)


# -------------------------
# Mixed precision (faster on GPU)
# -------------------------

scaler = torch.amp.GradScaler("cuda")


# -------------------------
# Training loop
# -------------------------

train_losses = []
val_losses = []
best_val_acc = 0

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in trainloader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # automatic mixed precision
        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    scheduler.step()

    train_loss = running_loss / len(trainloader)
    train_acc = 100 * correct / total

    train_losses.append(train_loss)


    # ----- validation -----

    model.eval()

    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in valloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= len(valloader)
    val_acc = 100 * val_correct / val_total

    val_losses.append(val_loss)

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}% "
          f"| Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}%")

    # save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_resnet18_task1_2.pth")


# -------------------------
# Plot loss curves
# -------------------------

plt.figure()

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")

plt.legend()

plt.savefig("loss_curve_task1_2.png")
plt.show()


# -------------------------
# Test evaluation
# -------------------------

model.load_state_dict(
    torch.load("best_resnet18_task1_2.pth", map_location=device)
)

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in testloader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


# overall accuracy

test_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))

print("\nTest Results")
print("Overall Accuracy:", test_acc)


# confusion matrix

cm = confusion_matrix(all_labels, all_preds)

print("\nConfusion Matrix:")
print(cm)


# per-class accuracy

class_accuracy = cm.diagonal() / cm.sum(axis=1)

for i, name in enumerate(trainset.classes):
    print(f"{name} Accuracy: {class_accuracy[i]*100:.2f}%")


# -------------------------
# Show some misclassified examples
# -------------------------

print("\nShowing failure cases...")

fail_images = []
fail_preds = []
fail_labels = []

with torch.no_grad():

    for images, labels in testloader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        wrong = predicted != labels

        for i in range(len(images)):
            if wrong[i]:
                fail_images.append(images[i].cpu())
                fail_preds.append(predicted[i].cpu())
                fail_labels.append(labels[i].cpu())

        if len(fail_images) >= 6:
            break


plt.figure(figsize=(12,6))

for i in range(6):

    img = fail_images[i]

    # reverse normalization for visualization
    img = img * torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    img = img + torch.tensor([0.485,0.456,0.406]).view(3,1,1)

    img = img.permute(1,2,0).numpy()

    plt.subplot(2,3,i+1)
    plt.imshow(img)
    plt.title(f"Pred: {trainset.classes[fail_preds[i]]}\nTrue: {trainset.classes[fail_labels[i]]}")
    plt.axis("off")

plt.show()
