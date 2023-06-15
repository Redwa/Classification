import numpy as np
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Step 1: Preprocess the data
comments = data['comment']
labels = data['label']

# Train Word2Vec model on the comments
sentences = [comment.split() for comment in comments]
word2vec_model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)

# Convert comments to average Word2Vec embeddings
X = np.zeros((len(comments), 100))
for i, comment in enumerate(sentences):
    comment_vectors = [word2vec_model[word] for word in comment if word in word2vec_model]
    if comment_vectors:
        X[i] = np.mean(comment_vectors, axis=0)

# Step 2: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Step 3: Train the classifier
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)

# Step 4: Evaluate the classifier
y_pred = svm_classifier.predict(X_test)
print(classification_report(y_test, y_pred))

# ----------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Step 2: Preprocess the data
comments = data['comment']
labels = data['label']

# Tokenize the comments
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokenized_comments = [tokenizer.encode(comment, add_special_tokens=True) for comment in comments]

# Pad and truncate the tokenized comments to a fixed length
max_length = max(len(comment) for comment in tokenized_comments)
padded_comments = [comment + [0] * (max_length - len(comment)) for comment in tokenized_comments]
attention_masks = [[1] * len(comment) + [0] * (max_length - len(comment)) for comment in tokenized_comments]

# Convert the data to PyTorch tensors
input_ids = torch.tensor(padded_comments)
attention_masks = torch.tensor(attention_masks)
labels = torch.tensor(labels)

# Step 3: Split the data
input_ids_train, input_ids_test, attention_masks_train, attention_masks_test, labels_train, labels_test = train_test_split(
    input_ids, attention_masks, labels, test_size=0.2, random_state=42
)

# Step 4: Load and fine-tune the DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=80)
model.train()

# Step 5: Train the classifier
optimizer = optim.Adam(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()

batch_size = 16
num_epochs = 5

for epoch in range(num_epochs):
    for i in range(0, len(input_ids_train), batch_size):
        batch_input_ids = input_ids_train[i:i+batch_size]
        batch_attention_masks = attention_masks_train[i:i+batch_size]
        batch_labels = labels_train[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(batch_input_ids, attention_mask=batch_attention_masks, labels=batch_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Step 6: Evaluate the classifier
model.eval()

with torch.no_grad():
    outputs = model(input_ids_test, attention_mask=attention_masks_test)
    logits = outputs.logits
    predicted_labels = np.argmax(logits.detach().numpy(), axis=1)

# Calculate evaluation metrics
from sklearn.metrics import classification_report
print(classification_report(labels_test, predicted_labels))

