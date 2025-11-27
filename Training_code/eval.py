import os
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AdamW

df = pd.read_csv('/cmlscratch/vinayakd/Workspace/DS_Project/shuffled_combined_dataset.csv')
df = pd.read_csv('/cmlscratch/vinayakd/Workspace/DS_Project/processed_dialseg711_test.csv')

print(df.shape)
print(df['topic_shift'].value_counts())

def get_boundaries(labels):
    # Convert sequence labels to boundaries
    boundaries = np.diff(labels, prepend=labels[0])
    return (boundaries != 0).astype(int)

tmp = df['topic_shift'].values
boundaries = get_boundaries(tmp)

print(df.shape[0]/(2*boundaries.sum()))

# exit()

# Step 1: Dataset
class TopicShiftDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Extract conversation history and new message
        row = self.data.iloc[index]
        history = self.get_history(row['conversation_id'], row['sub_conversation_id'], row['turn_id'])
        new_message = row['message']
        label = row['topic_shift']  # 0 or 1
        
        # Construct input
        input_text = f"History: {history} New message: {new_message}"
        encoded = self.tokenizer(
            input_text, 
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

    def get_history(self, conversation_id, sub_conversation_id, turn_id):
        # Filter conversation history up to the current turn_id
        subset = self.data[
            (self.data['conversation_id'] == conversation_id) &
            (self.data['sub_conversation_id'] == sub_conversation_id) &
            (self.data['turn_id'] < turn_id)
        ]
        return " ".join(subset['message'].tolist())

# Step 2: Model Definition
class TopicShiftClassifier(torch.nn.Module):
    def __init__(self, pretrained_model_name, num_labels):
        super(TopicShiftClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)

        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token representation
        logits = self.classifier(cls_output)
        return logits

# Step 3: Eval Code
def calculate_metrics(true_labels, pred_labels):
    """
    Calculates Pk, WindowDiff (WD), and Mean Absolute Error (MAE).
    """
    def get_boundaries(labels):
        # Convert sequence labels to boundaries
        boundaries = np.diff(labels, prepend=labels[0])
        return (boundaries != 0).astype(int)

    def pk(true_boundaries, pred_boundaries, k):
        # Pk metric implementation
        errors = 0
        for i in range(len(true_boundaries) - k):
            true_seg = true_boundaries[i:i+k].sum() > 0
            pred_seg = pred_boundaries[i:i+k].sum() > 0
            if true_seg != pred_seg:
                errors += 1
        return errors / (len(true_boundaries) - k)

    def window_diff(true_boundaries, pred_boundaries, k):
        # WindowDiff metric implementation
        errors = 0
        for i in range(len(true_boundaries) - k):
            true_count = true_boundaries[i:i+k].sum()
            pred_count = pred_boundaries[i:i+k].sum()
            if true_count != pred_count:
                errors += 1
        return errors / (len(true_boundaries) - k)
    
    def mae(true_shifts, pred_shifts):
        # Check if either array is empty
        if true_shifts.size == 0 or pred_shifts.size == 0:
            return len(true_shifts) + len(pred_shifts)
        
        # Calculate MAE using the minimum absolute difference for each true shift
        errors = [min(abs(ts - ps) for ps in pred_shifts) for ts in true_shifts]
        return np.mean(errors)

    # Convert labels to boundary format
    true_boundaries = get_boundaries(true_labels)
    pred_boundaries = get_boundaries(pred_labels)

    # Extract true and predicted topic shift positions
    true_shift_positions = np.where(true_boundaries == 1)[0]
    pred_shift_positions = np.where(pred_boundaries == 1)[0]

    # Define the window size for Pk and WD
    # k = int(0.5 * len(true_boundaries))  # Adjust window size as needed

    # print(true_boundaries.sum(), len(true_boundaries))
    # k = len(true_boundaries)/(2*true_boundaries.sum())
    k = 2

    # Calculate metrics
    pk_score = pk(true_boundaries, pred_boundaries, k)
    wd_score = window_diff(true_boundaries, pred_boundaries, k)
    mae_score = mae(true_shift_positions, pred_shift_positions)

    return pk_score, wd_score, mae_score

def evaluate_model_with_new_metrics(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Collect predictions and labels
            # preds = torch.argmax(outputs, dim=1)
            # print(preds.cpu().tolist())
            # all_preds.extend(preds.cpu().tolist())
            # all_labels.extend(labels.cpu().tolist())

            probs = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
            threshold = 0.63  # Set your desired threshold
            preds = (probs[:, 1] > threshold).long()  # Index 1 corresponds to "Topic Shift"
            # print(preds.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            # if (idx+1)%20 == 0:
            #     break

    avg_loss = total_loss / len(dataloader)

    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=["No Shift", "Topic Shift"], digits=4)

    # Calculate additional metrics
    pk_score, wd_score, mae_score = calculate_metrics(all_labels, all_preds)
    metrics_report = {
        "Pk": pk_score,
        "WD": wd_score,
        "MAE": mae_score
    }

    return avg_loss, report, metrics_report

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Collect predictions and labels
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader)

    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=["No Shift", "Topic Shift"], digits=4)
    # print(report)

    return avg_loss, report

def train_model(model, train_dataloader, eval_dataloader,  optimizer, criterion, device, epoch):
    
    # eval_loss, report = evaluate_model(model, eval_dataloader, criterion, device)
    # print(f"Eval loss: {eval_loss:.4f}\n\nClassification Report:\n\n{report}")

    eval_loss, report, metrics_report = evaluate_model_with_new_metrics(model, eval_dataloader, criterion, device)
    print(f"Eval Loss: {eval_loss:.4f}\n")
    print(f"Classification Report:\n{report}")
    print(f"Additional Metrics:\nPk: {metrics_report['Pk']:.4f}, WD: {metrics_report['WD']:.4f}, MAE: {metrics_report['MAE']:.4f}")

# Step 5: Main Function
def main(dataframe, pretrained_model_name="microsoft/deberta-v3-large", num_epochs=10, batch_size=128, max_length=512):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use all GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = TopicShiftClassifier(pretrained_model_name, num_labels=2)
        model = torch.nn.DataParallel(model)
    else:
        model = TopicShiftClassifier(pretrained_model_name, num_labels=2)

    model = TopicShiftClassifier(pretrained_model_name, num_labels=2)
    new_state_dict = {}
    state_dict = torch.load('./saved_deberta_FT_models/model_epoch_10.pth')
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")  # Remove "module." prefix
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    

    # Tokenizer and Dataset
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    dataset = TopicShiftDataset(dataframe, tokenizer, max_length)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Split dataset into train and eval (80/20 split)
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Model and optimizer
    # model = TopicShiftClassifier(pretrained_model_name, num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, no_deprecation_warning=True)

    # class_weights = compute_class_weight(class_weight='balanced', classes=[0, 1], y=train_labels)
    # class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Pass the class_weights to the loss function
    # criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    criterion = torch.nn.CrossEntropyLoss()

    os.makedirs("./saved_deberta_FT_models", exist_ok=True)

    # Training loop
    # for epoch in range(num_epochs):
    train_model(model, train_dataloader, eval_dataloader, optimizer, criterion, device, 0)

    return model

# Example Usage:
# Assuming `df` is your DataFrame
trained_model = main(df)
