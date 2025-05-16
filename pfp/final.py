import pandas as pd
from transformers import BertTokenizer, BertModel
import torch

# Load ProtBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)  # Ensures sequences are not lowercased
model = BertModel.from_pretrained('Rostlab/prot_bert')

# Function to get embeddings for a protein sequence
def get_protbert_embeddings(sequence):
    # Ensure the sequence is properly formatted (uppercase amino acids with no spaces)
    sequence = " ".join(list(sequence))  # Insert spaces between each amino acid
    inputs = tokenizer(sequence, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling to get fixed-size embeddings for the whole sequence
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Load your CSV file that contains the amino acid sequences
# Assume the column containing sequences is named 'Sequence'
data = pd.read_csv('D:\KATHIRAVAN\SEM5\Project\fold\proteinseq.csv')  # Replace with your actual file path

# Prepare to store embeddings
embeddings = []

# Iterate over all sequences and generate embeddings
for seq in data[' Sequence']:
    embedding = get_protbert_embeddings(seq)
    print(seq)

    embeddings.append(embedding[0])  # Append the first (and only) embedding

# Convert the embeddings list to a DataFrame
embeddings_df = pd.DataFrame(embeddings)

# Save the embeddings to a CSV file
embeddings_df.to_csv('protein_embed.csv', index=False)  # Replace with desired output path

print(f"Protein embeddings saved to 'protein_embeddings.csv'.")
