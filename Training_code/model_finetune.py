import os
import torch
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AdamW

df = pd.read_csv('/cmlscratch/vinayakd/Workspace/DS_Project/shuffled_combined_dataset.csv')
# print(df.info())

vals = df[['turn_id', 'topic_shift']].values
topic_shifts = []
for ii in vals:
    if ii[0] == 0 and ii[1] == 1:
        topic_shifts.append(0)
    else:
        topic_shifts.append(ii[1])

df['topic_shift'] = topic_shifts

print(df.shape)
print(df['topic_shift'].value_counts())

def check_violations(df):
    violations = []

    # Group by conversation_id
    grouped = df.groupby('conversation_id')
    for convo_id, group in grouped:
        group = group.sort_values(by='turn_id')
        
        # Rule 1: Check if turn_id starts at 0 and increments continuously
        turn_ids = group['turn_id'].tolist()
        # if turn_ids != list(range(len(turn_ids))):
        #     violations.append(f"Violation in conversation_id {convo_id}: turn_id sequence is incorrect.")
        
        # Rule 2: Check if topic_shift == 1 only when sub_conversation_id changes
        sub_convo_ids = group['sub_conversation_id'].tolist()
        topic_shifts = group['topic_shift'].tolist()
        for i in range(1, len(sub_convo_ids)):
            if sub_convo_ids[i] != sub_convo_ids[i-1] and topic_shifts[i] != 1:
                violations.append(
                    f"Violation in conversation_id {convo_id} at turn_id {group.iloc[i]['turn_id']}: "
                    f"sub_conversation_id changed but topic_shift != 1."
                )
            if sub_convo_ids[i] == sub_convo_ids[i-1] and topic_shifts[i] == 1:
                violations.append(
                    f"Violation in conversation_id {convo_id} at turn_id {group.iloc[i]['turn_id']}: "
                    f"sub_conversation_id did not change but topic_shift == 1."
                )

    return violations

# Check for violations
violations = check_violations(df)

# Output the violations
if violations:
    print("Violations found:")
    for v in violations:
        print(v)
else:
    print("No violations found!")

exit()

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
        history = self.get_history(row['conversation_id'], row['sub_conversation_id'], row['turn_id'], row['topic_shift'])
        new_message = row['message']
        label = row['topic_shift']  # 0 or 1
        
        # Construct input
        input_text = f"History: {history} New message: {new_message}"
        if label == 1:
            # print(row['conversation_id'], row['sub_conversation_id'], row['turn_id'])
            print(history)
            # print(input_text, row['turn_id'])
            print('\n\n----------------------------------------\n\n')
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

    def get_history(self, conversation_id, sub_conversation_id, turn_id, topic_shift):
        # Filter conversation history up to the current turn_id
        # print(self.data.shape)
        subset = self.data[
            (self.data['conversation_id'] == conversation_id) &
            (self.data['sub_conversation_id'] == sub_conversation_id) &
            (self.data['turn_id'] < turn_id)
        ]
        if topic_shift:
            dd1 = self.data[self.data['conversation_id'] == conversation_id]
            dd = dd1[dd1['turn_id'] < turn_id]
            print(dd1)
            print('-------------')
            print(dd)
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
    
def focal_loss(inputs, targets, alpha=[0.25, 0.75], gamma=2):
    ce_loss = F.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-ce_loss)  # Probability of the correct class

    alpha_tensor = torch.tensor(alpha, device=inputs.device)
    alpha_class = alpha_tensor[targets]

    focal_loss = alpha_class * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()

# Step 3: Training Loop
def train_model(model, train_dataloader, eval_dataloader,  optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for idx, batch in enumerate(tqdm(train_dataloader)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # loss = criterion(outputs, labels)
        loss = focal_loss(outputs, labels, alpha=[0.25, 0.75], gamma=2)
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (idx+1)%50==0:
            print(loss.item())
        
        if (idx+1)%100==0:
            eval_loss, report = evaluate_model(model, eval_dataloader, criterion, device)
            print(f"Eval loss: {eval_loss:.4f}\n\nClassification Report:\n\n{report}")

    # torch.save(model.state_dict(), os.path.join("./saved_deberta_FT_models", f"model_epoch_{epoch + 1}.pth"))

    avg_loss = total_loss / len(train_dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

# Step 4: Evaluation Loop
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

    # os.makedirs("./saved_deberta_FT_models", exist_ok=True)

    # Training loop
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_model(model, train_dataloader, eval_dataloader, optimizer, criterion, device, epoch)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

    return model

# Example Usage:
# Assuming `df` is your DataFrame
trained_model = main(df)
