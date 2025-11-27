# import pandas as pd

# data = pd.read_csv('/cmlscratch/vinayakd/Workspace/DS_Project/shuffled_combined_dataset.csv')
# # data.head(30)

# from tqdm import tqdm

# def create_topic_coherence_pairs(df):
#     positive_pairs = []
#     negative_pairs = []
    
#     # Iterate through each conversation
#     for convo_id in tqdm(df['conversation_id'].unique()):
#         convo_data = df[df['conversation_id'] == convo_id]
#         sub_convos = convo_data['sub_conversation_id'].unique()
        
#         # Generate positive pairs
#         for sub_convo_id in sub_convos:
#             sub_convo_data = convo_data[convo_data['sub_conversation_id'] == sub_convo_id]
#             messages = sub_convo_data['message'].tolist()
            
#             for i in range(len(messages)):
#                 for j in range(i+1, len(messages)):
#                     positive_pairs.append((messages[i], messages[j]))
        
#         # Generate negative pairs
#         for i in range(len(sub_convos)):
#             for j in range(i+1, len(sub_convos)):
#                 sub_convo_1 = convo_data[convo_data['sub_conversation_id'] == sub_convos[i]]['message'].tolist()
#                 sub_convo_2 = convo_data[convo_data['sub_conversation_id'] == sub_convos[j]]['message'].tolist()
                
#                 for msg1 in sub_convo_1:
#                     for msg2 in sub_convo_2:
#                         negative_pairs.append((msg1, msg2))
    
#     return positive_pairs, negative_pairs

# positive_pairs, negative_pairs = create_topic_coherence_pairs(data)

# import random

# random.shuffle(positive_pairs)
# random.shuffle(negative_pairs)

# # Extract the required number of samples
# sampled_positive_pairs = positive_pairs[:80000]
# sampled_negative_pairs = negative_pairs[:80000]

# import pandas as pd
# import json

# def save_pairs(positive_pairs, negative_pairs, output_format="csv"):
#     """
#     Saves positive and negative pairs in the desired format (CSV or JSON).
#     """
#     # Combine the pairs with labels
#     positive_data = [{"message1": p[0], "message2": p[1], "label": "positive"} for p in positive_pairs]
#     negative_data = [{"message1": n[0], "message2": n[1], "label": "negative"} for n in negative_pairs]
    
#     # Combine all data
#     all_pairs = positive_data + negative_data
    
#     # Save as CSV
#     if output_format == "csv":
#         df = pd.DataFrame(all_pairs)
#         df.to_csv("topic_coherence_pairs.csv", index=False)
#         print(f"Saved pairs to 'topic_coherence_pairs.csv'")
#         return df
    
#     # Save as JSON
#     elif output_format == "json":
#         with open("topic_coherence_pairs.json", "w") as json_file:
#             json.dump(all_pairs, json_file, indent=4)
#         print(f"Saved pairs to 'topic_coherence_pairs.json'")
#     else:
#         raise ValueError("Unsupported format. Please choose 'csv' or 'json'.")

# # Save the positive and negative pairs
# df = save_pairs(positive_pairs, negative_pairs, output_format="csv")  # Change to "json" if preferred

## --------------------------------------------------------------------------------------------------
# Let's build the revised implementation from scratch as described.
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define Dataset
class ContrastiveDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=256):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        inputs1 = self.tokenizer(
            pair["message1"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs2 = self.tokenizer(
            pair["message2"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        label = 1 if pair["label"] == "positive" else 0
        return {
            "input_ids1": inputs1["input_ids"].squeeze(0),
            "attention_mask1": inputs1["attention_mask"].squeeze(0),
            "input_ids2": inputs2["input_ids"].squeeze(0),
            "attention_mask2": inputs2["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float),
        }


# Load Dataset
data_path = "/cmlscratch/vinayakd/Workspace/DS_Project/topic_coherence_pairs.csv"  # Adjust path as necessary
pairs_df = pd.read_csv(data_path)

print(pairs_df.shape)
print(pairs_df['label'].value_counts())

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
dataset = ContrastiveDataset(pairs_df.to_dict("records"), tokenizer)
dataloader = DataLoader(dataset, batch_size=24, shuffle=True)

# Define Model
class ContrastiveDeberta(nn.Module):
    def __init__(self, model_name, projection_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.projection = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, projection_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
        projected = self.projection(cls_embedding)
        return F.normalize(projected, p=2, dim=1)  # L2 normalization

def contrastive_loss(embeddings1, embeddings2, labels, margin=1.5, distance_metric="cosine"):
    """
    Contrastive loss function for pre-defined positive and negative pairs.
    
    Args:
        embeddings1: Tensor of shape (batch_size, embedding_dim) - Anchor embeddings.
        embeddings2: Tensor of shape (batch_size, embedding_dim) - Positive/Negative embeddings.
        labels: Tensor of shape (batch_size,) - 1 for positive pairs, 0 for negative pairs.
        margin: Float, the margin for negative pairs.
        distance_metric: String, "euclidean" or "cosine" to determine the distance metric.
    
    Returns:
        Loss value (scalar).
    """
    # Compute distances
    if distance_metric == "euclidean":
        distances = torch.norm(embeddings1 - embeddings2, dim=1)  # Euclidean distance
    elif distance_metric == "cosine":
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        distances = 1 - torch.sum(embeddings1 * embeddings2, dim=1)  # Cosine distance
    else:
        raise ValueError("Unsupported distance metric. Use 'euclidean' or 'cosine'.")

    # Compute positive loss: y_i * D(e1, e2)^2
    positive_loss = labels * (distances ** 2) * 2
    print(f"POS LOSS : {positive_loss}")

    # Compute negative loss: (1 - y_i) * max(0, m - D(e1, e2))^2
    # negative_loss = (1 - labels) * F.relu(margin - distances) ** 2
    negative_loss = (1 - labels) * torch.log(1 + torch.exp(margin - distances))
    # negative_l = (1 - labels) * (distances ** 2)

    # print(f"NEG LOSS : {negative_l}")
    print(f"Labels : {labels}")

    # Combine losses and average
    loss = (positive_loss + negative_loss).sum() #* 1000
    print(loss)
    return loss, positive_loss, negative_loss


# Initialize Model and Optimizer
model = ContrastiveDeberta("microsoft/deberta-v3-large").to(device)
state_dict = torch.load('/cmlscratch/vinayakd/Workspace/DS_Project/deberta-v3-large-contrastive-pt1.pth')
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace("module.", "")  # Remove "module." prefix
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict, strict=True)
# model.load_state_dict(state_dict, strict=True)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
    
optimizer = AdamW(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=10)  # Smooth learning rate
print_interval = 50

# Training Loop
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):
        # Load inputs and labels
        input_ids1 = batch["input_ids1"].to(device)
        attention_mask1 = batch["attention_mask1"].to(device)
        input_ids2 = batch["input_ids2"].to(device)
        attention_mask2 = batch["attention_mask2"].to(device)
        labels = batch["label"].to(device)

        # Compute embeddings
        embeddings1 = model(input_ids=input_ids1, attention_mask=attention_mask1)
        embeddings2 = model(input_ids=input_ids2, attention_mask=attention_mask2)

        # Compute loss
        # loss = nt_xent_loss(embeddings1, embeddings2, labels)
        loss, pos_loss, neg_loss = contrastive_loss(embeddings1, embeddings2, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # if (step + 1) % print_interval == 0:
        #     for name, param in model.named_parameters():
        #         if param.requires_grad and param.grad is not None:
        #             print(f"Layer: {name}, Gradient Mean: {param.grad.abs().mean().item()}, Gradient Std: {param.grad.abs().std().item()}")
        #         elif param.grad is None:
        #             print(f"Layer: {name}, Gradient: None")

        optimizer.step()
        total_loss += loss.item()

        if (step + 1) % print_interval == 0:
            print(f"Step {step + 1}/{len(dataloader)}, Loss: {total_loss / (step+1):.4f}")

        if (step + 1) % print_interval*10 == 0:
            print(f"POS LOSS : {pos_loss}")
            print(f"NEG LOSS : {neg_loss}")

        break
    # torch.save(model.state_dict(), f'deberta-v3-large-contrastive-pt{epoch}.pth')
    break

    scheduler.step()
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(dataloader):.4f}")
