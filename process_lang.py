import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json
import numpy as np

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

voting_data_path = 'data/UN_DATA.csv'
embeddings_npy_path = 'data/embeddings.npy'  # Change the file extension to .npy

embeddings = np.zeros((7855, 384))

# Read in the data
df = pd.read_csv(voting_data_path)
titles = df['Title']

for i, t in tqdm(enumerate(titles)):
    embedding = model.encode(t)  # Encoding the current title
    embeddings[i] = embedding

# Save the embeddings as a .npy file
np.save(embeddings_npy_path, embeddings)

# Define the path to the embeddings .npy file
embeddings_npy_path = 'data/embeddings.npy'

# Load the embeddings from the .npy file
loaded_embeddings = np.load(embeddings_npy_path)

# Check the shape and content of the loaded embeddings
print("Shape of loaded embeddings:", loaded_embeddings.shape)
