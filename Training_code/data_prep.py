# import pandas as pd

# # Load the datasets
# dataset1 = pd.read_csv('/cmlscratch/vinayakd/Workspace/DS_Project/processed_TIAGE.csv')  # Replace with your file paths
# dataset1['turn_id'] = dataset1['message_id']
# dataset1 = dataset1.drop(columns=['agent', 'message_id'])
# print("Dataset 1 shape and columns:", dataset1.shape, dataset1.columns)

# dataset2 = pd.read_csv('/cmlscratch/vinayakd/Workspace/DS_Project/topical_chat_processed_train.csv')
# dataset2 = dataset2.drop(columns=['Unnamed: 0', 'agent'])
# print("Dataset 2 shape and columns:", dataset2.shape, dataset2.columns)

# dataset3 = pd.read_csv('/cmlscratch/vinayakd/Workspace/DS_Project/processed_dialseg711_test.csv')
# print("Dataset 3 shape and columns:", dataset3.shape, dataset3.columns)


# # print(dataset1['conversation_id'].max(), dataset2['conversation_id'].max(), dataset3['conversation_id'].max())
# # print(dataset1['conversation_id'].min(), dataset2['conversation_id'].min(), dataset3['conversation_id'].min())

# total_convos = dataset1['conversation_id'].max() + dataset2['conversation_id'].max() + dataset3['conversation_id'].max()
# # print(total_convos)

# # Reassign conversation_id from scratch for each dataset
# dataset2['conversation_id'] = [i+dataset1['conversation_id'].max() for i in dataset2['conversation_id'].values]
# dataset3['conversation_id'] = [i+dataset2['conversation_id'].max() for i in dataset3['conversation_id'].values]

# # Combine the datasets vertically (stacking rows)
# combined_dataset = pd.concat([dataset1, dataset2, dataset3], ignore_index=True)

# # Display the combined dataset to verify everything looks correct
# print("Combined dataset shape and columns:", combined_dataset.shape, combined_dataset.columns)

# # print(combined_dataset['conversation_id'].max())

# # Optionally, save to a new file
# combined_dataset.to_csv('original_combined_dataset.csv')
# exit()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from transformers import MarianMTModel, MarianTokenizer
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import os

from transformers import MarianMTModel, MarianTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
import json


# Load the combined dataset
data_path = "new_combined_dataset.csv"
df = pd.read_csv(data_path, index_col=0)

# Inspect the dataset
print(df.head(20))
print(df.shape)

# Ensure sorting by conversation_id and turn_id
df = df.sort_values(by=["conversation_id", "turn_id"]).reset_index(drop=True)

# Save the cleaned dataset
# df.to_csv("sorted_dialogues.csv", index=False)

## ------------------------------------------------------------------------------------------------------------

#### Augmentation

# Dialogue Shuffling

import pandas as pd
import random

def shuffle_dialogues(df, num_samples=5000):
    
    shuffled_data = []
    unique_convos = df['conversation_id'].unique()
    new_convo_id = max(df['conversation_id']) + 1  # Start a new conversation ID that is unique

    for _ in range(num_samples):
        # Randomly select two conversation IDs
        convo_ids = random.sample(list(unique_convos), 2)

        # Extract the first conversation and its sub-conversations
        convo_1 = df[df['conversation_id'] == convo_ids[0]]
        sub_convos_1 = convo_1['sub_conversation_id'].unique()
        selected_sub_convo_1 = convo_1[convo_1['sub_conversation_id'] == random.choice(sub_convos_1)]

        # Ensure sub-conversation length is between 2 to 6
        if len(selected_sub_convo_1) < 2:
            continue

        # Ensure even number of turns
        if len(selected_sub_convo_1) % 2 != 0:
            selected_sub_convo_1 = selected_sub_convo_1.iloc[:-1]  # Remove the last turn if odd

        max_length = min(len(selected_sub_convo_1), 6)
        selected_sub_convo_1 = selected_sub_convo_1.iloc[:random.choice([2, 4, 6])]

        # Repeat for the second conversation
        convo_2 = df[df['conversation_id'] == convo_ids[1]]
        sub_convos_2 = convo_2['sub_conversation_id'].unique()
        selected_sub_convo_2 = convo_2[convo_2['sub_conversation_id'] == random.choice(sub_convos_2)]

        # Ensure sub-conversation length is between 2 to 6
        if len(selected_sub_convo_2) < 2:
            continue

        if len(selected_sub_convo_2) % 2 != 0:
            selected_sub_convo_2 = selected_sub_convo_2.iloc[:-1]  # Remove the last turn if odd

        max_length = min(len(selected_sub_convo_2), 6)
        selected_sub_convo_2 = selected_sub_convo_2.iloc[:random.choice([2, 4, 6])]

        # Update metadata
        selected_sub_convo_1['sub_conversation_id'] = 0
        selected_sub_convo_2['sub_conversation_id'] = 1
        # selected_sub_convo_2['topic_shift'].iloc[0] = 1  # Mark the shift for the first row of the second sub-convo
        selected_sub_convo_2.loc[selected_sub_convo_2.index[0], 'topic_shift'] = 1

        # Update conversation ID for the combined segments
        selected_sub_convo_1['conversation_id'] = new_convo_id
        selected_sub_convo_2['conversation_id'] = new_convo_id

        # Concatenate the two selected sub-conversations
        combined = pd.concat([selected_sub_convo_1, selected_sub_convo_2]).reset_index(drop=True)
        combined['turn_id'] = range(1, len(combined) + 1)

        # Append the new combined conversation
        shuffled_data.append(combined)

        # Increment the new conversation ID for the next shuffled conversation
        new_convo_id += 1

    # Combine all into a single DataFrame
    final_df = pd.concat(shuffled_data, ignore_index=True)
    return final_df

# Apply the function
shuffled_data = shuffle_dialogues(df, num_samples=4000)
print(shuffled_data.head(30))

shuffled_combined_dataset = pd.concat([df, shuffled_data], ignore_index=True)
shuffled_combined_dataset.to_csv("shuffled_combined_dataset.csv", index=False)

print(df['topic_shift'].value_counts())
print(shuffled_combined_dataset['topic_shift'].value_counts())

# exit()

# Paraphrasing and Back-Translation

# Load back-translation models (English -> French -> English)
# Check if GPU is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Load the back-translation models (English <-> French)
# bt_model_name = 'Helsinki-NLP/opus-mt-en-fr'
# bt_tokenizer = MarianTokenizer.from_pretrained(bt_model_name)
# bt_model = MarianMTModel.from_pretrained(bt_model_name).to(device)

# bt_model_rev_name = 'Helsinki-NLP/opus-mt-fr-en'
# bt_tokenizer_rev = MarianTokenizer.from_pretrained(bt_model_rev_name)
# bt_model_rev = MarianMTModel.from_pretrained(bt_model_rev_name).to(device)

# # Load the paraphrasing model
# paraphrase_model_name = 'tuner007/pegasus_paraphrase'
# paraphrase_tokenizer = AutoTokenizer.from_pretrained(paraphrase_model_name)
# paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained(paraphrase_model_name).to(device)

# # Define the back-translation function with GPU acceleration
# def back_translate(sentences, src_tokenizer, src_model, tgt_tokenizer, tgt_model, batch_size=16):
#     rephrased_sentences = []
#     for i in tqdm(range(0, len(sentences), batch_size), desc="Back-Translating"):
#         batch = sentences[i:i + batch_size]
#         # English to French
#         encoded = src_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
#         translated_tokens = src_model.generate(**encoded)
#         french_texts = [src_tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]

#         # French to English
#         encoded = tgt_tokenizer(french_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
#         back_translated_tokens = tgt_model.generate(**encoded)
#         back_translations = [tgt_tokenizer.decode(t, skip_special_tokens=True) for t in back_translated_tokens]
        
#         rephrased_sentences.extend(back_translations)
#     return rephrased_sentences

# # Define the paraphrasing function with GPU acceleration
# def paraphrase(sentences, tokenizer, model, batch_size=8, num_return_sequences=1):
#     rephrased_sentences = []
#     for i in tqdm(range(0, len(sentences), batch_size), desc="Paraphrasing"):
#         batch = sentences[i:i + batch_size]
#         encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
#         generated = model.generate(
#             **encoded,
#             max_length=128,
#             num_beams=5,
#             num_return_sequences=num_return_sequences,
#             temperature=1.5
#         )
#         paraphrases = tokenizer.batch_decode(generated, skip_special_tokens=True)
#         # Take the first paraphrase for simplicity
#         rephrased_sentences.extend(paraphrases[:len(batch)])
#     return rephrased_sentences

# # Define the full rephrasing pipeline with GPU acceleration
# def rephrase_dataframe(df, message_column="message"):
#     # Step 1: Back-translate all messages
#     messages = df[message_column].tolist()
#     # back_translated = back_translate(messages, bt_tokenizer, bt_model, bt_tokenizer_rev, bt_model_rev)
    
#     # Step 2: Paraphrase the back-translated messages
#     paraphrased = paraphrase(messages, paraphrase_tokenizer, paraphrase_model)
    
#     # Add the paraphrased column to the DataFrame
#     df["paraphrased_message"] = paraphrased
#     return df

# # Apply the pipeline on the dataset
# shuffled_combined_dataset = pd.read_csv("shuffled_combined_dataset.csv")
# paraphrased_df = rephrase_dataframe(shuffled_combined_dataset)

# Save the paraphrased DataFrame
# paraphrased_df.to_csv("paraphrased_shuffled_combined_dataset.csv", index=False)

# exit()
## ------------------------------------------------------------------------------------------------------------

#### Create Pairs for Contrastive Learning

# Topic Coherence Contrastive Learning
def create_topic_coherence_pairs(df):
    positive_pairs = []
    negative_pairs = []
    
    # Iterate through each conversation
    for convo_id in df['conversation_id'].unique():
        convo_data = df[df['conversation_id'] == convo_id]
        sub_convos = convo_data['sub_conversation_id'].unique()
        
        # Generate positive pairs
        for sub_convo_id in sub_convos:
            sub_convo_data = convo_data[convo_data['sub_conversation_id'] == sub_convo_id]
            messages = sub_convo_data['message'].tolist()
            
            for i in range(len(messages)):
                for j in range(i+1, len(messages)):
                    positive_pairs.append((messages[i], messages[j]))
        
        # Generate negative pairs
        for i in range(len(sub_convos)):
            for j in range(i+1, len(sub_convos)):
                sub_convo_1 = convo_data[convo_data['sub_conversation_id'] == sub_convos[i]]['message'].tolist()
                sub_convo_2 = convo_data[convo_data['sub_conversation_id'] == sub_convos[j]]['message'].tolist()
                
                for msg1 in sub_convo_1:
                    for msg2 in sub_convo_2:
                        negative_pairs.append((msg1, msg2))
    
    return positive_pairs, negative_pairs

positive_pairs, negative_pairs = create_topic_coherence_pairs(shuffled_combined_dataset)

print(positive_pairs[:10])

# exit()

def save_pairs(positive_pairs, negative_pairs, output_format="csv"):
    """
    Saves positive and negative pairs in the desired format (CSV or JSON).
    """
    # Combine the pairs with labels
    positive_data = [{"message1": p[0], "message2": p[1], "label": "positive"} for p in positive_pairs]
    negative_data = [{"message1": n[0], "message2": n[1], "label": "negative"} for n in negative_pairs]
    
    # Combine all data
    all_pairs = positive_data + negative_data
    
    # Save as CSV
    if output_format == "csv":
        df = pd.DataFrame(all_pairs)
        df.to_csv("topic_coherence_pairs.csv", index=False)
        print(f"Saved pairs to 'topic_coherence_pairs.csv'")
    
    # Save as JSON
    elif output_format == "json":
        with open("topic_coherence_pairs.json", "w") as json_file:
            json.dump(all_pairs, json_file, indent=4)
        print(f"Saved pairs to 'topic_coherence_pairs.json'")
    else:
        raise ValueError("Unsupported format. Please choose 'csv' or 'json'.")

# Save the positive and negative pairs
save_pairs(positive_pairs, negative_pairs, output_format="csv")  # Change to "json" if preferred


# Shift-Aware Contrastive Learning
def create_shift_aware_pairs(df):
    anchor_positive_pairs = []
    anchor_negative_pairs = []
    
    for convo_id in df['conversation_id'].unique():
        convo_data = df[df['conversation_id'] == convo_id]
        
        for i, row in convo_data.iterrows():
            anchor = row['message']
            current_sub_convo_id = row['sub_conversation_id']
            
            # Find positives (same sub_conversation_id)
            positives = convo_data[convo_data['sub_conversation_id'] == current_sub_convo_id]['message'].tolist()
            positives = [msg for msg in positives if msg != anchor]
            
            # Find negatives (following topic shift)
            negatives = convo_data[convo_data['turn_id'] > row['turn_id']]
            negatives = negatives[negatives['topic_shift'] == 1]['message'].tolist()
            
            for positive in positives:
                anchor_positive_pairs.append((anchor, positive))
            
            for negative in negatives:
                anchor_negative_pairs.append((anchor, negative))
    
    return anchor_positive_pairs, anchor_negative_pairs

anchor_positive_pairs, anchor_negative_pairs = create_shift_aware_pairs(df)

