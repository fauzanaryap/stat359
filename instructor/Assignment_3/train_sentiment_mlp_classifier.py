# ========== Imports ==========
import numpy as np
import pandas as pd
import datasets
import gensim.downloader as api
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

print("\n========== Loading Dataset ==========")
# ========== Load Dataset ==========
dataset = datasets.load_dataset('financial_phrasebank', 'sentences_50agree', trust_remote_code=True)
print("Dataset loaded. Example:", dataset['train'][0])

print("\n========== Preparing DataFrame ==========")
data = pd.DataFrame(dataset['train'])
data['text_label'] = data['label'].apply(lambda x: 'positive' if x == 2 else 'neutral' if x == 1 else 'negative')
print(f"DataFrame shape: {data.shape}")

texts = data['sentence'].tolist()
labels = data['label'].values

def get_device():
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

device = get_device()
print(f"Using device: {device}")

# ========== Load FastText ==========
print('loading fasttext')
fasttext = api.load("fasttext-wiki-news-subwords-300")
EMBED_DIM = 300

def tokenize(text):
    return text.lower().split()

def sentence_to_avg_vector(text):
    tokens = tokenize(text)
    vectors = [fasttext[t] for t in tokens if t in fasttext]
    if len(vectors) == 0:
        return np.zeros(EMBED_DIM)
    return np.mean(vectors, axis=0)

print('encoding sentences')

X = np.array([sentence_to_avg_vector(text) for text in tqdm(texts)])
y = labels

print('splitting data')

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size = 0.15, stratify = y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size = 0.15, stratify = y_trainval, random_state=42)

print('pytorch dataset')

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(SimpleDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader   = DataLoader(SimpleDataset(X_val, y_val), batch_size=32, shuffle=False)
test_loader  = DataLoader(SimpleDataset(X_test, y_test), batch_size=32, shuffle=False)

print('defining model')

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)
    
num_classes = len(set(labels))
model = MLPClassifier(EMBED_DIM, num_classes).to(device)

print('training setup')

class_weights = compute_class_weight(
    class_weight = 'balanced',
    classes = np.unique(y_train),
    y = y_train
)

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', patience = 3)

print('training loop')

num_epochs = 30
best_val_f1 = 0.0
train_loss_history = []
val_loss_history = []
train_f1_history = []
val_f1_history = []
train_acc_history = []
val_acc_history = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    train_preds = []
    train_labels = []

    for inputs, labels_batch in train_loader:
        inputs, labels_batch = inputs.to(device), labels_batch.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)

        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels_batch.cpu().numpy())

    train_loss = running_loss / len(train_loader.dataset)
    train_f1 = f1_score(train_labels, train_preds, average='macro')
    train_acc = np.mean(np.array(train_labels) == np.array(train_preds))

    train_loss_history.append(train_loss)
    train_f1_history.append(train_f1)
    train_acc_history.append(train_acc)

    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for inputs, labels_batch in val_loader:
            inputs, labels_batch = inputs.to(device), labels_batch.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels_batch)

            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)

            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels_batch.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    val_f1 = f1_score(val_labels, val_preds, average='macro')
    val_acc = np.mean(np.array(val_labels) == np.array(val_preds))

    val_loss_history.append(val_loss)
    val_f1_history.append(val_f1)
    val_acc_history.append(val_acc)

    scheduler.step(val_f1)

    print(f"Epoch {epoch+1}: Train F1: {train_f1:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Epoch {epoch+1}: Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "best_mlp.pth")

print("training complete!")

model.load_state_dict(torch.load("best_mlp.pth"))
model.eval()

test_preds = []
test_labels = []

with torch.no_grad():
    for inputs, labels_batch in test_loader:
        inputs, labels_batch = inputs.to(device), labels_batch.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels_batch.cpu().numpy())

test_acc = np.mean(np.array(test_labels) == np.array(test_preds))
test_f1_macro = f1_score(test_labels, test_preds, average='macro')

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1 Macro: {test_f1_macro:.4f}")

print("\nClassification Report:")
print(classification_report(test_labels, test_preds))

cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot = True , fmt = 'd', cmap = 'Blues')
plt.title('Confusion Matrix (MLP)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('mlp_confusion_matrix.png')
plt.show()

epochs = range(1, num_epochs + 1)

plt.figure(figsize=(18,5))

# ---- Loss Plot ----
plt.subplot(1,3,1)
plt.plot(epochs, train_loss_history, label='Train Loss')
plt.plot(epochs, val_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()

# ---- F1 Plot ----
plt.subplot(1,3,2)
plt.plot(epochs, train_f1_history, label='Train F1 (Macro)')
plt.plot(epochs, val_f1_history, label='Validation F1 (Macro)')
plt.axhline(y=test_f1_macro, linestyle='--', label=f'Test F1 ({test_f1_macro:.4f})')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('F1 Score vs Epoch')
plt.legend()

# ---- Accuracy Plot ----
plt.subplot(1,3,3)
plt.plot(epochs, train_acc_history, label='Train Accuracy')
plt.plot(epochs, val_acc_history, label='Validation Accuracy')
plt.axhline(y=test_acc, linestyle='--', label=f'Test Acc ({test_acc:.4f})')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epoch')
plt.legend()

plt.tight_layout()
plt.savefig("mlp_training_curves_with_test.png")
plt.show()


print('script complete!')