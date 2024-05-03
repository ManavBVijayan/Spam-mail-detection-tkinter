import string
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pickle
from imblearn.over_sampling import SMOTE


def clean_text(message):
    message_not_punc = []  # Message without punctuation
    i = 0
    for punctuation in message:
        if punctuation not in string.punctuation:
            message_not_punc.append(punctuation)
    # Join words again to form the string.
    message_not_punc = ''.join(message_not_punc)

    # Remove any stopwords for message_not_punc
    message_clean = list(message_not_punc.split(" "))
    while i <= len(message_clean):
        for mess in message_clean:
            if mess.lower() in stopwords.words('english'):
                message_clean.remove(mess)
        i = i + 1
    return message_clean


data = pd.read_csv("Data/mail_data.csv")
data.columns = ['label', 'messages']

# Data cleaning
data["length"] = data["messages"].apply(len)
data['messages'] = data['messages'].apply(clean_text)

# Feature extraction
vectorization = CountVectorizer(analyzer=clean_text)
X = vectorization.fit_transform(data['messages'])

tfidf_transformer = TfidfTransformer().fit(X)
X_tfidf = tfidf_transformer.transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, data['label'], test_size=0.30, random_state=50)
oversample = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = oversample.fit_resample(X_train, y_train)
# Train model
model = SVC(kernel='linear', C=1.0)
model.fit(X_train_balanced, y_train_balanced)

# Evaluate model
predictions = model.predict(X_test)
print('predicted', predictions)
print(classification_report(y_test, predictions))

# Save the model for future use
with open('spam_classifier.sav', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")
