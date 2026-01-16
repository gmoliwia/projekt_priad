import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


nltk.download('stopwords')
df = pd.read_csv('emails.csv')
print("Kolumny w pliku:", df.columns)
def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-z]', ' ', text) #usuwanie znaków specjalnych i cyfr
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['text_clean'] = df['text'].apply(clean_text)
#Podział na dane
X = df['text_clean']
y = df['spam'] # 0 = ham, 1 = spam
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

#Przewidywanie i Ewaluacja
y_pred = model.predict(X_test_tfidf)

print("\n--- WYNIKI MODELU ---")
print(f"Dokładność (Accuracy): {accuracy_score(y_test, y_pred):.2f}")
print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred))

# Macierz pomyłek (Confusion Matrix)
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Przewidziane')
plt.ylabel('Rzeczywiste')
plt.title('Macierz Pomyłek')
plt.show()

#Test na własnym przykładzie
def predict_spam(text):
    cleaned = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)
    return "SPAM" if prediction[0] == 1 else "Normalna wiadomość (HAM)"

print("\n--- TEST NA NOWEJ WIADOMOŚCI ---")
test_email = "Congratulations! You have won a 1000 dollar gift card. Click here to claim now."
print(f"Treść: {test_email}")
print(f"Wynik: {predict_spam(test_email)}")