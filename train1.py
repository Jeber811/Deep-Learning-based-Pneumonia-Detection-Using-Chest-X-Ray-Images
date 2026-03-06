# Train ResNet18 from scratch for pneumonia classification
# Includes augmentation, validation monitoring, model checkpointing,
# loss curves, test evaluation, and examples of misclassified images.

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# Set random seeds so training results are reproducible
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)


# Data augmentation for training images
# Validation and test images only receive resizing + normalization
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# Dataset paths (Google Colab structure)
data_dir = "/content/chest_xray"
train_dir = data_dir + "/train"
val_dir   = data_dir + "/val"
test_dir  = data_dir + "/test"


# Load datasets using ImageFolder
trainset = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
valset   = torchvision.datasets.ImageFolder(val_dir, transform=val_transform)
testset  = torchvision.datasets.ImageFolder(test_dir, transform=val_transform)


# Data loaders
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True
)

valloader = torch.utils.data.DataLoader(
    valset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
)

print("Train distribution:", np.bincount(trainset.targets))
print("Validation distribution:", np.bincount(valset.targets))


# Training configuration
EPOCHS = 5
BATCH_SIZE = 16
LR = 0.0003

print("\nTraining configuration")
print("----------------------")
print("Architecture: ResNet18")
print("Optimizer: AdamW")
print("Learning rate:", LR)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Loss: CrossEntropy + label smoothing")
print("Scheduler: Cosine Annealing")
print("Mixed precision: enabled\n")


# Handle class imbalance using inverse frequency weighting
class_counts = np.bincount(trainset.targets)
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
class_weights = class_weights.to(device)


# Build the model (ResNet18 without pretrained weights)
model = torchvision.models.resnet18(weights=None)

# Replace final classification layer
model.fc = nn.Linear(model.fc.in_features, len(trainset.classes))

model = model.to(device)


# Loss function with label smoothing
criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1
)


# Optimizer and learning rate scheduler
optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=10
)


# Mixed precision training (faster on GPUs)
scaler = torch.amp.GradScaler("cuda")


train_losses = []
val_losses = []
best_val_acc = 0


# -------------------
# Training loop
# -------------------

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in trainloader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

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


    # Validation pass (no gradients needed)
    model.eval()

    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in valloader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(valloader)
    val_acc = 100 * val_correct / val_total

    val_losses.append(val_loss)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}% "
        f"| Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}%"
    )

    # Save model if validation accuracy improves
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "/content/best_resnet18_model.pth")


# -------------------
# Plot training curves
# -------------------

plt.figure()

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")

plt.legend()

plt.savefig("/content/loss_curve.png")
plt.show()


# -------------------
# Test evaluation
# -------------------

model.load_state_dict(
    torch.load("/content/best_resnet18_model.pth", map_location=device)
)

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in testloader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


# Overall test accuracy
test_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))

print("\nTest accuracy:", test_acc)


# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)

print("\nConfusion matrix:")
print(cm)


# Per-class accuracy
class_accuracy = cm.diagonal() / cm.sum(axis=1)

for i, name in enumerate(trainset.classes):
    print(f"{name} accuracy: {class_accuracy[i]*100:.2f}%")


# -------------------
# Show misclassified images
# -------------------

print("\nDisplaying some misclassified examples...")

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

    img = fail_images[i] * 0.5 + 0.5
    img = img.permute(1,2,0)

    plt.subplot(2,3,i+1)
    plt.imshow(img)
    plt.title(
        f"Pred: {trainset.classes[fail_preds[i]]}\n"
        f"True: {trainset.classes[fail_labels[i]]}"
    )
    plt.axis("off")

plt.show()
