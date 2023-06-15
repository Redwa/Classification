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
