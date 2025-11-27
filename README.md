# Topic-Shift-Detection-in-Multi-Turn-Dialogs



## 📋 Project Overview
This project addresses the challenge of identifying topic shifts—moments when a conversation deviates from its prior subject—in multi-turn dialogues. Detecting these shifts is fundamental for chatbot development, dialogue summarization, and conversational analytics.

Current methods often struggle with dialogue complexity, contextual dependencies, and a lack of robust benchmark datasets. This work introduces a new benchmark dataset and a DeBERTa-v3-large based Transformer architecture designed to effectively encode turn-level semantic features and aggregate context.

## 🚀 Key Contributions
* **New Benchmark Dataset:** A unified dataset of ~115k samples created by harmonizing Topical Chat, TIAGE, and Dialseg711, focusing on diversity and realism.
* **Novel Training Paradigm:** Combines **Contrastive Learning** with supervised fine-tuning to better capture semantic transitions.
* **State-of-the-Art Performance:** Outperforms strong baselines (including T5 and BERT) on metrics such as $P_k$, WindowDiff ($WD$), and $F1$.

---

## 💾 Dataset
We curated and preprocessed three primary datasets into a unified tabular structure to resolve inconsistencies and class imbalances.

### Data Sources & Processing
1.  **Topical Chat:** Filtered to max 20 turns; removed conversations with >5 topic shifts.
2.  **TIAGE:** Parsed into tabular format with turn-level annotations.
3.  **Dialseg711:** Capped at 20 turns and 5 shifts per conversation.

### Synthetic Augmentation
To address class imbalance (originally 17.4% positive examples), we generated synthetic topic shifts by merging conversations, increasing the dataset to approximately **115k samples** with 24.3% positive examples.

| Dataset | Original Format | Post-Filter Rows | Avg Topic Shifts |
| :--- | :--- | :--- | :--- |
| **Topical Chat** | JSON (Persona) | 76,424 | 4.2 |
| **TIAGE** | Text Files | 6,239 | 4.6 |
| **Dialseg711** | JSONL | 19,351 | 3.8 |
| **Synthetic** | - | 12,921 | 3.9 |

---

## 🧠 Methodology

### 1. Contrastive Pre-Training
We utilized contrastive learning to teach DeBERTa v3 to map semantically similar messages closer in embedding space and dissimilar messages farther apart.

**Objective Function:**
The model optimizes the Contrastive Loss $L$:
$$L=-\\frac{1}{N}\\sum_{i=1}^{N}log\\frac{exp(sim(h_{i},h_{j})/\\tau)}{\\sum_{k=1}^{2N}exp(sim(h_{i},h_{k})/\\tau)}$$
*Where $h_i, h_j$ are positive pair embeddings, and $\\tau$ is the temperature parameter.*

**Strategies:**
* **Topic Coherence:** Positive pairs are generated from the same sub-conversation; negative pairs from different sub-conversations.
* **Shift-Aware:** Focuses on message pairs around topic shifts to improve sensitivity to context changes.

### 2. Model Architecture (Fine-Tuning)
* **Backbone:** DeBERTa v3-large.
* **Input Format:** Pairs of consecutive turns are tokenized as `[CLS] message_1 [SEP] message_2 [SEP]`.
* **Freezing Strategy:** Most layers are frozen to leverage pre-learned representations; only the final classification layer is trained.
* **Context Management:** Uses a sliding window approach, truncating history only if it exceeds 512 tokens.

---

## 📊 Results & Performance

The model was evaluated using **$F1$-Score**, **Accuracy**, **$P_k$** (lower is better), and **WindowDiff ($WD$)** (lower is better).

### Quantitative Comparison
Our model achieved **92.7% accuracy** on the validation set and established new benchmarks on TIAGE and Dialseg711.

| Model | Dataset | $P_k$ $\\downarrow$ | $WD$ $\\downarrow$ | $F1$ $\\uparrow$ |
| :--- | :--- | :--- | :--- | :--- |
| **Our Model** | **TIAGE** | **0.1768** | **0.2186** | **0.7731** |
| RoBERTa | TIAGE | 0.265 | 0.287 | 0.572 |
| **Our Model** | **Dialseg711** | **0.1399** | **0.1683** | **0.8139** |
| BERT | Dialseg711 | 0.214 | 0.225 | 0.725 |

* **TIAGE Improvement:** Outperformed RoBERTa and TextSegDial significantly.
* **Dialseg711 Improvement:** Reduced segmentation errors ($WD$) by 23.8% compared to traditional architectures.

---

## 🛠️ Usage & Configuration

### Hyperparameters
* **Batch Size:** 128 (using 4x RTX A6000 GPUs).
* **Learning Rate:** Initial 1e-4, adjusted to 2e-5 to prevent overfitting.
* **Optimizer:** AdamW with weight decay.
* **Class Weights:** Applied to handle imbalance (Topic Shift vs. No Shift).

### Code Snippet: Class Weight Calculation
To handle dataset imbalance during fine-tuning:
```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# y_train is the array of topic shift labels (0 or 1)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)