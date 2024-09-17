import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pickle

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the Siamese Network class
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.feature_extractor = models.resnet18(weights=None)
        self.feature_extractor.fc = nn.Sequential(
            nn.Linear(self.feature_extractor.fc.in_features, 512),
            nn.ReLU(inplace=True)
        )

    def forward_once(self, x):
        output = self.feature_extractor(x)
        return output

# Load the trained model
@st.cache_resource
def load_model(model_path):
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Load embeddings from file
@st.cache_resource
def load_embeddings(filename='embeddings.pkl'):
    with open(filename, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

# Function to compute embedding for an input image
def compute_embedding(model, img, transform):
    image = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model.forward_once(image).cpu().numpy().flatten()
    
    return embedding

# Function to find the closest match
def find_closest_match(embedding, embeddings):
    min_dist = float('inf')
    closest_person = None
    
    for img_path, (stored_embedding, person_name) in embeddings.items():
        dist = np.linalg.norm(embedding - stored_embedding)  # Euclidean distance
        if dist < min_dist:
            min_dist = dist
            closest_person = person_name
    
    return closest_person, min_dist

# Function to compare two images and compute dissimilarity score
def compare_images(model, img1, img2, transform):
    # Compute embeddings for both images
    embedding1 = compute_embedding(model, img1, transform)
    embedding2 = compute_embedding(model, img2, transform)
    
    # Calculate Euclidean distance (dissimilarity score)
    euclidean_distance = np.linalg.norm(embedding1 - embedding2)
    
    return euclidean_distance

# Define the transform (same as used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the model and embeddings
model_path = 'siamese_network.pth'  # Ensure this is the correct path to your saved model
model = load_model(model_path)
embeddings = load_embeddings('embeddings.pkl')

# Streamlit Application
st.title("Siamese Network Application")

# Sidebar options
option = st.sidebar.selectbox(
    "Choose an option",
    ("Compare Images", "Identify Person")
)

if option == "Compare Images":
    st.header("Compare Two Images")

    # Upload two images
    img1 = st.file_uploader("Upload the first image", type=["jpg", "jpeg", "png"], key="img1")
    img2 = st.file_uploader("Upload the second image", type=["jpg", "jpeg", "png"], key="img2")

    if img1 is not None and img2 is not None:
        # Convert the uploaded files to PIL images
        image1 = Image.open(img1).convert('RGB')
        image2 = Image.open(img2).convert('RGB')

        # Display the images
        st.image([image1, image2], caption=["First Image", "Second Image"], width=300)

        # Compare the images
        if st.button("Compare Images"):
            distance = compare_images(model, image1, image2, transform)
            st.write(f"Dissimilarity Score (Euclidean Distance) between the images: {distance:.4f}")

elif option == "Identify Person":
    st.header("Identify a Person from Image")

    # Upload an image
    img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="img")

    if img is not None:
        # Convert the uploaded file to a PIL image
        image = Image.open(img).convert('RGB')

        # Display the image
        st.image(image, caption="Uploaded Image", width=300)

        # Identify the person
        if st.button("Identify Person"):
            input_embedding = compute_embedding(model, image, transform)
            person_name, distance = find_closest_match(input_embedding, embeddings)
            st.write(f"The person in the image is likely: {person_name} with a distance of {distance:.4f}")
