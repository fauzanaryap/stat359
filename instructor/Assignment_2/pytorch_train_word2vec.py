import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 512  # change it to fit your memory constraints, e.g., 256, 128 if you run out of memory
EPOCHS = 7
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive

# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):
    
    def __init__ (self, pairs_array):
        self.pairs = pairs_array
    
    def __len__(self):
        return self.pairs.shape[0]
    
    def __getitem__(self, idx):
        center = self.pairs[idx, 0]
        context = self.pairs[idx, 1]

        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long)


# Simple Skip-gram Module
class Word2Vec(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, center_words, context_words):
        
        v = self.in_embed(center_words)

        if context_words.dim() == 1:
            u = self.out_embed(context_words)
            return torch.sum(v * u, dim = 1)
    
        u = self.out_embed(context_words)
        return torch.sum(v.unsqueeze(1) * u, dim = 2)

    def get_embeddings(self):
        return self.in_embed.weight.detach().cpu().numpy()


# Load processed data

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / 'processed_data.pkl'

with open(DATA_PATH, 'rb') as f:
    data = pickle.load(f)

word2idx = data['word2idx']
idx2word = data['idx2word']
counter = data['counter']
skipgram_df = data['skipgram_df']

vocab_size = len(word2idx)

pairs = skipgram_df[['center', 'context']].to_numpy()

# Precompute negative sampling distribution below

counts = torch.zeros(vocab_size, dtype=torch.float)

for word, c in counter.items():
    counts[word2idx[word]] = c

neg_dist = counts ** 0.75
neg_dist = neg_dist / neg_dist.sum()

# Device selection: CUDA > MPS > CPU

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")

neg_dist = neg_dist.to(device)

# Dataset and DataLoader

dataset = SkipGramDataset(pairs)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model, Loss, Optimizer

model = Word2Vec(vocab_size, EMBEDDING_DIM).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def make_targets(center, context, vocab_size):
    
    batch_size = center.size(0)
    
    neg_context = torch.multinomial(neg_dist, batch_size * NEGATIVE_SAMPLES, replacement=True).view(batch_size, NEGATIVE_SAMPLES)
    neg_context = neg_context.to(center.device)

    mask = neg_context.eq(context.unsqueeze(1))

    while mask.any(): 
        neg_context[mask] = torch.multinomial(neg_dist, mask.sum().item(), replacement=True)
        mask = neg_context.eq(context.unsqueeze(1))

    pos_labels = torch.ones(batch_size, dtype=torch.float).to(device)
    neg_labels = torch.zeros((batch_size, NEGATIVE_SAMPLES), dtype=torch.float).to(device)
    
    return context, neg_context, pos_labels, neg_labels

print("Number of skip-gram pairs:", len(pairs))
print("Batches per epoch:", len(pairs) // BATCH_SIZE)

# Training loop
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0

    for center, context in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):

        center = center.to(device)
        context = context.to(device)

        pos_context, neg_context, pos_labels, neg_labels = make_targets(center, context, vocab_size)

        pos_scores = model(center, pos_context)
        neg_scores = model(center, neg_context)

        loss_pos = criterion(pos_scores, pos_labels)
        loss_neg = criterion(neg_scores, neg_labels)

        loss = loss_pos + loss_neg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()   
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")


# Save embeddings and mappings

embeddings = model.get_embeddings()

with open('word2vec_embeddings.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddings, 'word2idx': data['word2idx'], 'idx2word': data['idx2word']}, f)
print("Embeddings saved to word2vec_embeddings.pkl")
