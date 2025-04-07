import os
import re
import string
from typing import Tuple
import joblib
from nltk.corpus import stopwords
import joblib
from typing import Tuple
from sklearn.base import BaseEstimator

from nltk.tokenize import word_tokenize



MODEL_PATH = r"/app/Model" # r"C:\Users\Keshav\Desktop\Assignment 4\app\Model" 


# cleaning the text
def preprocess_text(text):
    # 1. Case Folding
    text = text.lower()

    # 2. Remove Punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # 3. Tokenization
    tokens = word_tokenize(text)

    # 4. Remove Stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # 5. Remove word of length 1
    tokens = [word for word in tokens if len(word) > 1]

    # 6. Remove Extra Whitespace and join tokens back to string
    cleaned_text = ' '.join(tokens)

    return cleaned_text



def load_model_vectoriser(model_path: str) -> Tuple[BaseEstimator, object]:
    model = joblib.load(os.path.join(model_path, "model.pkl"))
    vectorizer = joblib.load(os.path.join(model_path, "vectorizer.pkl"))
    return model, vectorizer

# model, vectoriser = load_model_vectoriser(MODEL_PATH)

def score(text, model, vectoriser, threshold=0.5):
    text = preprocess_text(text)
    x = vectoriser.transform([text])
    probabilities = model.predict_proba(x)
    propensity = probabilities[0][1]
    prediction = propensity >= threshold
    return bool(prediction), round(propensity,3)



def main():
    model, vectoriser = load_model_vectoriser(MODEL_PATH)
    texts = [
        "The scholarship amount has been deposited in your account. Thank you.",
        "Free money, give a miscall on 1234567890",
        "You account has been hacked. Give me money.",
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's.",
        "This email is spam.",
        "you text me back",
        "Congratulations! You have won a free iPhone.",
        "Congratulations! You've won a  gift card! Click here to claim your prize",
        "Hi Alice, we found your resume on our site and would like to offer you a work-from-home position. Reply with your email for more details!"

    ]
    for text in texts:
        print(text)
        prediction, propensity = score(text, model, vectoriser)
        print(prediction)
        print("Spam" if prediction else "Not spam", end = " || ")
        print(f"Prob[text is spam] = {propensity}")
        print()


if __name__ == "__main__":
    main()





