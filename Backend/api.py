import pandas as pd
import chromadb
import torch
from transformers import BertTokenizer, BertModel
from flask import Flask, request, jsonify, redirect
from flasgger import Swagger

# Define file paths and ChromaDB settings
chroma_db_path = "D:/project/Chromadb"  # Directory for ChromaDB persistence
collection_name = "recipes"

# Initialize ChromaDB client
client = chromadb.PersistentClient(
    path=chroma_db_path
)

# Load the existing collection
collection = client.get_collection(name=collection_name)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Initialize Flask app and Swagger
app = Flask(__name__)
swagger = Swagger(app)

def embed_text(text):
    """Embed the input text using BERT."""
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Get the mean of the last hidden state to obtain a single vector representation
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

def find_most_similar(user_input):
    """Find the most similar document in the ChromaDB collection."""
    # Embed the user input
    user_embedding = embed_text(user_input)

    # Query ChromaDB for the most similar document using the embedded input
    results = collection.query(
        query_embeddings=[user_embedding],
        n_results=1  # We only want the most similar document
    )

    return results

@app.route('/')
def index():
    """Redirect root URL to Swagger UI."""
    return redirect('/apidocs/')

@app.route('/api/find_similar', methods=['POST'])
def find_similar():
    """API endpoint to find the most similar recipe based on user input.
    ---
    parameters:
      - name: query
        in: body
        type: string
        required: true
        description: The recipe query to find similar recipes.
    responses:
      200:
        description: A similar recipe found.
        schema:
          type: object
          properties:
            title:
              type: string
              example: "Spaghetti Carbonara"
            metadata:
              type: object
              example: {"ingredients": ["spaghetti", "eggs", "parmesan cheese", "pancetta"]}
      400:
        description: No query provided.
      404:
        description: No similar recipe found.
    """
    data = request.get_json()
    user_input = data.get('query')

    if not user_input:
        return jsonify({"error": "No query provided"}), 400

    similar_document = find_most_similar(user_input)

    if similar_document and similar_document['documents']:
        response = {
            "title": similar_document['documents'][0],
            "metadata": similar_document['metadatas'][0]
        }
        return jsonify(response), 200
    else:
        return jsonify({"message": "No similar recipe found."}), 404

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
