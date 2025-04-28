import json
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle


def main(file_name):
    with open(file_name, "r") as file:
        train_data = json.load(file)

    # Texte und Labels einsammeln
    texts = [text for key in train_data for text in train_data[key]]
    labels = [key for key in train_data for _ in train_data[key]]

    # Daten in Trainings- und Testdaten aufteilen
    texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2,                                                     random_state=42)

    # TF-IDF Vektorisierer initialisieren
    tfidf_vectorizer = TfidfVectorizer()

    # Den Vektorisierer an die Trainingsdaten anpassen und diese Texte in TF-IDF-Vektoren umwandeln
    x_train = tfidf_vectorizer.fit_transform(texts_train)
    x_test = tfidf_vectorizer.transform(texts_test)

    # Den Klassifizierer initialisieren und trainieren
    clf = SVC()
    clf.fit(x_train, labels_train)

    # Labels für das Testset vorhersagen
    labels_pred = clf.predict(x_test)

    # Die Akkuratheit berechnen
    accuracy = accuracy_score(labels_test, labels_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Das Modell speichern
    with open('hochzeit_intent.pkl', 'wb') as save_file:
        pickle.dump(clf, save_file)

    # Das Modell laden
    with open('hochzeit_intent.pkl', 'rb') as read_file:
        model = pickle.load(read_file)

    # Einen neuen Text für die Vorhersage vorbereiten
    new_text = ["Suchen Sie Leute?"]
    new_text_tfidf = tfidf_vectorizer.transform(new_text)

    # Das Label (Intent) für den neuen Text vorhersagen
    predicted_label = clf.predict(new_text_tfidf)[0]
    print(predicted_label)

if __name__ == '__main__':
    #try:
    main("train.json")
    #except FileNotFoundError as e:
        #print(e)
