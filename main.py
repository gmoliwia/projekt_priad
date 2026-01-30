import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Pobranie listy słów stopu (opcjonalnie, jeśli nie masz ich lokalnie)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 1. Wczytanie danych
df = pd.read_csv('emails.csv')

# 2. Wstępna analiza (EDA)
print("Pierwsze wiersze danych:")
print(df.head())
print(f"\nRozkład klas:\n{df['spam'].value_counts(normalize=True)}")

# Wizualizacja rozkładu
sns.countplot(x='spam', data=df)
plt.title('Rozkład wiadomości: 0 - Real, 1 - Spam')
plt.show()

# 3. Preprocessing tekstu
def clean_text(text):
    # Usunięcie słowa 'Subject: ' z początku
    text = re.sub(r'^Subject: ', '', text)
    # Usunięcie znaków interpunkcyjnych i cyfr, zamiana na małe litery
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    # Tokenizacja i usunięcie stop words
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 1]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(clean_text)

# 4. Podział na zbiór treningowy i testowy
X = df['clean_text']
y = df['spam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Wektoryzacja (Bag of Words) - zgodnie z PRiAD 7
vectorizer = CountVectorizer(max_features=3000) # Ograniczamy do 3000 najczęstszych słów
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Modelowanie (Naiwny Klasyfikator Bayesowski)
# Wybrany ze względu na wysoką skuteczność w tekstach
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 7. Ewaluacja modelu
y_pred = model.predict(X_test_vec)

print("\n--- RAPORT KLASYFIKACJI ---")
print(classification_report(y_test, y_pred))

# Macierz pomyłek
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Przewidywane')
plt.ylabel('Rzeczywiste')
plt.title('Macierz pomyłek (Confusion Matrix)')
plt.show()

print(f"Dokładność modelu (Accuracy): {accuracy_score(y_test, y_pred):.2%}")