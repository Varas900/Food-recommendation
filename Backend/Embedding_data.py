import os
import glob
import vaex
import pandas as pd
import dask.dataframe as dd
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

MAX_ROWS_PER_SESSION = 50000
BATCH_SIZE = 100  # Rows per batch
df = vaex.open('D:/project/actual dataset/recipes_data.csv')
df = df[['title', 'ingredients', 'directions', 'NER']]
df_pandas = df.to_pandas_df()

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def tokenize_text_batch(texts):
    tokens_list = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i + 32]
        tokens = tokenizer(batch.tolist(), padding=True, truncation=True, return_tensors='pt')
        tokens_list.append(tokens)
    return tokens_list

def get_bert_embeddings_batch(tokens_list):
    embeddings_list = []
    for tokens in tokens_list:
        with torch.no_grad():
            outputs = model(**tokens)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy().tolist()
            embeddings_list.extend(embeddings)
    return embeddings_list

def process_batch(text_series):
    tokens = tokenize_text_batch(text_series)
    embeddings = get_bert_embeddings_batch(tokens)
    return embeddings

def average_embeddings(row):
    """Calculate the average of the embeddings from the specified columns without reducing dimensions."""
    title_emb = np.array(row['title_embeddings'])
    
    # Initialize a list for valid embeddings
    valid_embeddings = [title_emb]

    # Get ingredient, direction, and NER embeddings
    if row['ingredients_embeddings']:
        valid_embeddings.extend([np.array(emb) for emb in row['ingredients_embeddings']])
    if row['directions_embeddings']:
        valid_embeddings.extend([np.array(emb) for emb in row['directions_embeddings']])
    if row['NER_embeddings']:
        valid_embeddings.extend([np.array(emb) for emb in row['NER_embeddings']])
    
    # Calculate the average only if there are valid embeddings
    if valid_embeddings:
        avg_emb = np.mean(valid_embeddings, axis=0)
    else:
        avg_emb = np.zeros(title_emb.shape)  # If no embeddings are available, return a zero vector of the same size

    return avg_emb

def process_and_save_batches(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    batches = [df[i:i + BATCH_SIZE] for i in range(0, len(df), BATCH_SIZE)]

    for idx, batch in enumerate(batches):
        batch_filename = os.path.join(output_dir, f'processed_batch_{idx + 1}.csv')
        if os.path.exists(batch_filename):
            continue

        ddf_batch = dd.from_pandas(batch, npartitions=1)
        ddf_batch['title_embeddings'] = ddf_batch['title'].map_partitions(process_batch, meta=('title_embeddings', 'object'))
        ddf_batch['ingredients_embeddings'] = ddf_batch['ingredients'].map_partitions(process_batch, meta=('ingredients_embeddings', 'object'))
        ddf_batch['directions_embeddings'] = ddf_batch['directions'].map_partitions(process_batch, meta=('directions_embeddings', 'object'))
        ddf_batch['NER_embeddings'] = ddf_batch['NER'].map_partitions(process_batch, meta=('NER_embeddings', 'object'))

        ddf_batch = ddf_batch.compute()

        # Calculate the average embeddings and create a new column
        ddf_batch['average_embeddings'] = ddf_batch.apply(average_embeddings, axis=1)

        # Select the relevant columns for saving
        embeddings_df_batch = ddf_batch[['title', 'ingredients', 'directions', 'NER', 'average_embeddings']]

        # Save to CSV
        embeddings_df_batch.to_csv(batch_filename, index=False)
        print(f'Saved {batch_filename} with {len(embeddings_df_batch)} rows.')

total_rows_to_process = min(MAX_ROWS_PER_SESSION, len(df_pandas))

process_and_save_batches(df_pandas[:total_rows_to_process], 'D:/project/Dataset/processed_batches')

def combine_csv_files(input_dir, output_dir, output_file='combined_processed_dataset.csv'):
    all_files = glob.glob(os.path.join(input_dir, 'processed_batch_*.csv'))
    combined_df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    combined_df.to_csv(os.path.join(output_dir, output_file), index=False)
    print(f'Combined all processed files into {os.path.join(output_dir, output_file)}')

input_directory = 'D:/project/Dataset/processed_batches'
output_directory = 'D:/project/Dataset'

combine_csv_files(input_directory, output_directory)
