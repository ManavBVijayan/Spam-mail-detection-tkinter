from tkinter import *
import pickle
from spam_classifier import vectorization, tfidf_transformer

# Load the trained model
with open('spam_classifier.sav', 'rb') as f:
    model = pickle.load(f)


# Function to preprocess email text and classify it
def classify_email(email_text):
    # Preprocess the email text
    email_transformed = vectorization.transform([email_text])
    email_tfidf = tfidf_transformer.transform(email_transformed)
    prediction = model.predict(email_tfidf)[0]
    result = "Spam" if prediction == 1 else "Ham"
    return result


def clear_entry():
    email_entry.delete("1.0", END)


def process_email():
    email_text = email_entry.get("1.0", END)
    result = classify_email(email_text)
    result_label.config(text=f"Email Classification: {result}")
    clear_entry()


# Initialize Tkinter window
window = Tk()
window.title("Email Spam Classifier")

# Email entry field
email_label = Label(window, text="Enter Email Text:")
email_label.pack()
email_entry = Text(window, height=10, width=50)
email_entry.pack()

# Classification button
classify_button = Button(window, text="Classify Email", command=process_email)
classify_button.pack()

# Result label
result_label = Label(window, text="")
result_label.pack()

# Run the main loop
window.mainloop()
