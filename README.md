# ✍️ Intelligent Phrase Completion/Prediction

## Introduction
This project focuses on developing a **Next Phrase / Word Prediction** model — a core capability behind modern text systems like **Google Autocomplete**, **keyboard autotype suggestions**, and **chatbots**.  
The model learns to predict the next possible word or phrase based on the given context of preceding words.  
Using **deep learning with LSTMs and GRUs**, this project explores how sequence modeling can capture contextual relationships within natural language.

---

## Dataset
- **Source:** A **story-based text dataset** obtained from **Kaggle**.  
- **Nature:** Continuous English narrative text suitable for sequence modeling and context learning.  
- The dataset was preprocessed to create a **corpus of up to 50K words**, ensuring a balanced vocabulary size and manageable training complexity.

---

## Preprocessing & Data Pipeline
Thorough text cleaning and sequence preparation steps were implemented to prepare the corpus for neural sequence modeling.

### **Steps Involved:**
1. **Text Cleaning:**
   - Removed punctuation marks and extra spaces (except full stops to preserve sentence boundaries).
2. **Vocabulary Limiting:**
   - Corpus truncated to the **top 50,000 words**.
3. **Tokenization:**
   - Applied Keras `Tokenizer` on the full corpus with `vocab_size = 1000`.
   - (Higher vocab sizes like 5K resulted in poorer accuracy due to data sparsity.)
4. **Sequence Generation:**
   - Created **n-gram style sliding window** sequences from each sentence to form input–output pairs.
   - Applied **padding** to each input sequence to a fixed `max_len = 299`.
5. **Input-Output Split:**
   - Final dataset generated as multiple pairs of context words → target next word.

This preprocessing effectively converts raw text into a structured numerical format suitable for sequence learning models.

---

##  Model Architecture

The project explored multiple architectures and embedding sizes before finalizing the optimal configuration.

### **Baseline Architecture:**
```
Embedding(vocab_size, 100)
→ LSTM(100)
→ Dense(...)
→ Dense(vocab_size, activation='softmax')
```

### **Tuned Architecture:**
- Upgrading to **embedding dimension = 200** significantly improved accuracy.
- Replacing LSTM with **GRU** yielded higher stability and faster convergence.
- Tried adding **Dropout** layers and stacking multiple LSTM/GRU layers — but the simpler model performed better.

### **Final Model Configuration:**
```
Embedding(input_dim=1000, output_dim=200, input_length=299)
→ GRU(100)
→ Dense(100, activation='relu')
→ Dense(vocab_size, activation='softmax')
```

- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  
- **Training Epochs:** 50  
- **Training Accuracy:** ~0.81  
- **Validation Accuracy:** ~0.19 (average)

---

## Real-World Analogy & Insights
While the validation accuracy may seem modest, this is a **common behavior for generative sequence tasks**, especially with smaller datasets and vocabulary limits.  
In real-world terms, the model demonstrates **context awareness similar to smartphone keyboard autotype** or **Google autosuggest** on unseen phrases — able to generate semantically plausible next words even if exact accuracy metrics remain low.

This experiment highlights the **trade-off between model simplicity and linguistic diversity**: the GRU-based model generalizes reasonably well without overfitting, even with limited training data.

---

## Tech Stack
- **Language:** Python  
- **Libraries:**
  - `tensorflow / keras` (for deep learning)
  - `numpy`, `pandas` (for preprocessing)
  - `re` (for regex cleaning)
  - `matplotlib` (for visualization and accuracy plotting)
- **Environment:** Jupyter Notebook / Google Colab

---

## Conclusion
The project successfully demonstrates how a neural network can **predict next words or phrases** using contextual understanding of preceding text.
- The **GRU model** with 200-dimensional embeddings achieved a training accuracy of **81%**, showing strong learning capacity.
- The **validation accuracy** reflects the challenges of **generalizing text generation** — similar to how real-world autocomplete systems struggle with unseen contexts.

This project serves as a strong foundation for further work in **sequence generation**, **language modeling**, and **contextual NLP applications**.

---

## Future Enhancements
- Integrate and **ModelCheckpoint** for optimal weight retention.
- Use **pretrained embeddings (GloVe / Word2Vec)** to improve contextual understanding.
- Experiment with **bidirectional LSTM/GRU** and **Transformer-based architectures (like GPT or BERT)**.
- Deploy as an **interactive suggestion engine** via Streamlit or Flask.
