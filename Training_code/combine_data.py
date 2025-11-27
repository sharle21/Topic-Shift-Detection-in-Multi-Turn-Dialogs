import pandas as pd
import numpy as np
import random

df1 = pd.read_csv('/cmlscratch/vinayakd/Workspace/DS_Project/topical_chat_processed_train.csv')
df1 = df1.drop(columns=['agent'])
topic_shift = [1 if i == 'True' else 0 for i in df1['topic_shift']]
df1['topic_shift'] = topic_shift
max_conv_id = np.max(df1['conversation_id'].values)

# print(df1.shape)
# print(df1.head())

df2 = pd.read_csv('/cmlscratch/vinayakd/Workspace/DS_Project/processed_TIAGE.csv')
df2['turn_id'] = df2['message_id']
conv_ids = [i+max_conv_id for i in df2['conversation_id']]
df2['conversation_id'] = conv_ids
df2 = df2.drop(columns=['agent', 'message_id'])
max_conv_id = np.max(df2['conversation_id'].values)

# print(df2.shape)
# print(df2.head())

df3 = pd.read_csv('/cmlscratch/vinayakd/Workspace/DS_Project/processed_dialseg711_test.csv')
conv_ids = [i+max_conv_id for i in df3['conversation_id']]
df3['conversation_id'] = conv_ids

# print(df3.shape)
# print(df3.head())

df_combined = pd.concat([df1, df2, df3], ignore_index=True)
df_combined.reset_index(inplace=True)

print(df_combined.shape)
# print(df_combined[-5:])

df_combined["sub_convo_change"] = df_combined["sub_conversation_id"].diff().ne(0)  # True if sub_convo_id changes
fault_condition_1 = (df_combined["sub_convo_change"] == True) & (df_combined["topic_shift"] != 1)
fault_condition_2 = (df_combined["sub_convo_change"] == False) & (df_combined["topic_shift"] == 1)

# Count faulty rows
faulty_rows_count_1 = df_combined[fault_condition_1].shape[0]
faulty_rows_count_2 = df_combined[fault_condition_2].shape[0]

# print(faulty_rows_count_1, faulty_rows_count_2)

df_combined.loc[fault_condition_1, "topic_shift"] = 1

df_combined["sub_convo_change"] = df_combined["sub_conversation_id"].diff().ne(0)  # True if sub_convo_id changes
fault_condition_1 = (df_combined["sub_convo_change"] == True) & (df_combined["topic_shift"] != 1)
fault_condition_2 = (df_combined["sub_convo_change"] == False) & (df_combined["topic_shift"] == 1)

# Verify the changes by checking the faulty conditions again
faulty_rows_count_1_after = df_combined[fault_condition_1].shape[0]
faulty_rows_count_2_after = df_combined[fault_condition_2].shape[0]

# print(faulty_rows_count_1_after, faulty_rows_count_2_after)
# df_combined.to_csv('/cmlscratch/vinayakd/Workspace/DS_Project/new_combined_dataset.csv')

# exit()
