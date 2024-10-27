import pandas as pd
import chromadb
import re
import ast
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
import numpy as np

# Define file paths
csv_file_path = "D:/project/Food recommendation//Backend/Dataset/combined_processed_dataset.csv"
chroma_db_path = "D:/project/Food recommendation//Backend/Chromadb"  # Directory for ChromaDB persistence

# Load data from CSV
data_df = pd.read_csv(csv_file_path)


# Initialize ChromaDB client with a custom persistence directory
client = chromadb.PersistentClient(
    path=chroma_db_path,
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)

# Create collection
collection_name = "recipes"
collection = client.create_collection(name=collection_name)

def normalize_id(title):
    return re.sub(r'[^a-zA-Z0-9]', '_', title)

for index, row in data_df.iterrows():
    doc_id = normalize_id(row["title"])

    # Convert the average_embeddings correctly
    # Assuming average_embeddings is a string representation of a NumPy array
    average_embeddings = np.fromstring(row["average_embeddings"].strip('[]'), sep=' ').tolist()  # Convert to list

    # Process other fields safely
    ingredients = ast.literal_eval(row["ingredients"])
    directions = ast.literal_eval(row["directions"])
    ner = ast.literal_eval(row["NER"])

    collection.add(
        documents=[row["title"]],
        embeddings=[average_embeddings],  # Ensure it's a list of 768 dimensions
        metadatas=[{
            "ingredients": str(ingredients),
            "directions": str(directions),
            "NER": str(ner)
        }],
        ids=[doc_id]
    )

print("Data successfully saved to ChromaDB.")

# Print the first row of the DataFrame
first_row = data_df.iloc[2]
print(first_row)
