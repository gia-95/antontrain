import torch
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import os
import json
import datetime


############################# PARAMETRI ############################

DIR_DATASET_TRAIN = "dataset_augm/train"
DIR_DATASET_VAL = "dataset_augm/val"
DIR_SAVED_MODEL = "modelli_resNet18"
DIR_HISTORIES = "histories"
num_epochs = 35
tipo_train = "casco_rota_bright"
now = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

######################################################################

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)

# Freeze all the pre-trained layers
for param in model.parameters():
    param.requires_grad = False

# Modify the last layer of the model
num_classes = 2 # replace with the number of classes in your dataset
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)


# Define the transformations to apply to the images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the train and validation datasets
train_dataset = ImageFolder(DIR_DATASET_TRAIN, transform=transform)
val_dataset = ImageFolder(DIR_DATASET_VAL, transform=transform)

print("\ntrain_dataset:",len(train_dataset), "frame\n")

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# Create data loaders for the train and validation datasets
batch_size = 64 if device == 'cuda' else 32
print("batch_size:", batch_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size , shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Variabili per salvataggio 
history = []

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    
    print("-------- TRAINING ----------")
    
    # Train the model for the specified number of epochs
    for epoch in range(num_epochs):
        # Set the model to train mode
        model.train()

        # Initialize the running loss and accuracy
        running_loss = 0.0
        running_corrects = 0

        # Iterate over the batches of the train loader
        for inputs, labels in train_loader:
            # Move the inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the optimizer gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()

            # Update the running loss and accuracy
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # Calculate the train loss and accuracy
        train_loss = running_loss / len(train_dataset)
        train_acc = running_corrects.double() / len(train_dataset)

        # Set the model to evaluation mode
        model.eval()

        # Initialize the running loss and accuracy
        running_loss = 0.0
        running_corrects = 0

        # Iterate over the batches of the validation loader
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Move the inputs and labels to the device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Update the running loss and accuracy
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        # Calculate the validation loss and accuracy
        val_loss = running_loss / len(val_dataset)
        val_acc = running_corrects.double() / len(val_dataset)

        # Print the epoch results
        print('Epoch [{}/{}], train loss: {:.4f}, train acc: {:.4f}, val loss: {:.4f}, val acc: {:.4f}'
              .format(epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))
        
        history.append((epoch+1, train_loss.item(), train_acc.item(), val_loss.item(), val_acc.item()))
        
    print(history)
    ### Save history
    try :
        with open(f'{DIR_HISTORIES}/hist_{tipo_train}_{num_epochs}eph_{now}.txt', 'w') as filehandle:
            json.dump(history, filehandle)
    except :  # noqa: E722
        print("Errore write-file!")
        


# Set the device
model.to(device)

# Fine-tune the last layer for a few epochs
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9)
train(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)

# Unfreeze all the layers and fine-tune the entire network for a few more epochs
# for param in model.parameters():
#     param.requires_grad = True
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)


### Save de model
model_path = os.path.join(DIR_SAVED_MODEL, f"model_{tipo_train}_{num_epochs}eph_{now}"+".pth")
torch.save(model, model_path)