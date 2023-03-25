from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
df=pd.read_csv('spam_ham_dataset.csv')
# Load the dataset
messages = [df['text']]  # list of text messages
labels = [df['label']]  # list of labels (0 for "ham", 1 for "spam")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(messages, labels, test_size=0.2, random_state=42)

# Extract features using the Bag of Words model
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train a Naive Bayes classifier on the training set
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

# Make predictions on the testing set and evaluate performance
y_pred = clf.predict(X_test_counts)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
