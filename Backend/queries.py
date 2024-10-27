import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to generate embedding using BERT
def generate_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Taking the mean of the last hidden state to get fixed-size embedding
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Define ChromaDB path
chroma_db_path = "D:/project/Chromadb"  # Directory for ChromaDB persistence

# Initialize ChromaDB client with the same persistence directory
client = chromadb.PersistentClient(
    path=chroma_db_path,
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)

# Specify the collection name
collection_name = "recipes"

# Retrieve the collection
collection = client.get_collection(name=collection_name)

# Get user input
user_input = input("Enter your query: ")

# Generate the embedding for the user input
query_embedding = generate_embedding(user_input)

# Query using the embedding to find the closest match
query_result = collection.query(
    query_embeddings=[query_embedding],
    n_results=1  # Get the closest match
)

# Check if any documents were found
if query_result['documents']:
    # Since we expect only one document, we can access it directly
    document_id = query_result['ids'][0]  # Accessing the ID
    title = query_result['documents'][0]  # Accessing the title
    metadata = query_result['metadatas'][0]  # Accessing metadata
    embedding = query_result['embeddings'][0]  # Accessing the embedding

    # Print all details of the retrieved document
    print("Document ID:", document_id)
    print("Title:", title)
    print("Metadata:", metadata)
    print("Embedding:", embedding)
else:
    print(f"No documents found for the query: {user_input}")
