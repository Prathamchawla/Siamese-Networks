import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
import random
import pickle  # To save embeddings

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset Class
class SiameseNetworkDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_pairs = []
        self.labels = []
        self._prepare_pairs()

    def _prepare_pairs(self):
        """Prepare pairs of images for training."""
        folders = os.listdir(self.root_dir)
        for i in range(len(folders)):
            folder1 = folders[i]
            folder1_path = os.path.join(self.root_dir, folder1)
            if not os.path.isdir(folder1_path):
                continue
            
            images1 = os.listdir(folder1_path)
            for image1 in images1:
                image1_path = os.path.join(folder1_path, image1)
                
                # Positive pair
                image2 = random.choice(images1)
                image2_path = os.path.join(folder1_path, image2)
                self.image_pairs.append((image1_path, image2_path))
                self.labels.append(1)
                
                # Negative pair
                j = random.choice([x for x in range(len(folders)) if x != i])
                folder2 = folders[j]
                folder2_path = os.path.join(self.root_dir, folder2)
                images2 = os.listdir(folder2_path)
                image2 = random.choice(images2)
                image2_path = os.path.join(folder2_path, image2)
                self.image_pairs.append((image1_path, image2_path))
                self.labels.append(0)

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]
        label = self.labels[idx]
        
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

# Transformations with Normalization for Pre-trained Model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization for pre-trained models
])

# Load the dataset
dataset = SiameseNetworkDataset(root_dir=r'C:\Users\spars\Desktop\Siamese Network Streamlit\Dataset', transform=transform)
dataloader = DataLoader(dataset, shuffle=True, batch_size=32)

# Define the Siamese Network using a Pre-trained Model
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.feature_extractor = models.resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Sequential(
            nn.Linear(self.feature_extractor.fc.in_features, 512),
            nn.ReLU(inplace=True)
        )

    def forward_once(self, x):
        output = self.feature_extractor(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

# Initialize model, loss, and optimizer
model = SiameseNetwork().to(device)
criterion = ContrastiveLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for i, (img1, img2, label) in enumerate(dataloader):
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        
        optimizer.zero_grad()
        output1, output2 = model(img1, img2)
        loss = criterion(output1, output2, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

# Save the trained model
torch.save(model.state_dict(), 'siamese_network.pth')
print("Model saved!")

# Function to extract and save embeddings for all images
def extract_and_save_embeddings(model, dataset_dir, transform, embedding_file='embeddings.pkl'):
    embeddings = {}
    model.eval()
    
    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                embedding = model.forward_once(image).cpu().numpy().flatten()
            
            embeddings[image_path] = (embedding, person_name)
    
    # Save embeddings to a file
    with open(embedding_file, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings saved to {embedding_file}")

# Extract and save embeddings for the dataset
extract_and_save_embeddings(model, r'C:\Users\spars\Desktop\Siamese Network Streamlit\Dataset', transform)

# Load the model for evaluation
def load_model(model_path):
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Function to compare two images and compute dissimilarity score
def compare_images(model, img_path1, img_path2, transform):
    img1 = Image.open(img_path1).convert('RGB')
    img2 = Image.open(img_path2).convert('RGB')

    if transform:
        img1 = transform(img1).unsqueeze(0).to(device)
        img2 = transform(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        output1, output2 = model(img1, img2)
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        print(f"Dissimilarity Score (Euclidean Distance) between the images: {euclidean_distance.item()}")

# Load the saved model
model = load_model('siamese_network.pth')

# Compare two images
img1_path = r'C:\Users\spars\Desktop\Siamese Network Streamlit\Dataset\Nipun\WhatsApp Image 2024-09-17 at 4.57.27 PM.jpeg'
img2_path = r'C:\Users\spars\Desktop\Siamese Network Streamlit\Dataset\Udit\WhatsApp Image 2024-09-17 at 4.54.12 PM.jpeg'
compare_images(model, img1_path, img2_path, transform)
