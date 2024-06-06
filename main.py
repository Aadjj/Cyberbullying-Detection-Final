import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

df_labelled_tweets = pd.read_csv('C:/Users/aadjj/Downloads/labeled_tweets.csv')
df_public = pd.read_csv('C:/Users/aadjj/Downloads/public_data_labeled.csv')
df_combined = pd.concat([df_labelled_tweets[['label', 'full_text']], df_public])

print("Before handling missing values")
df_combined['label'] = df_combined['label'].apply(lambda x: 1 if x == 'Offensive' else 0)
df_combined['full_text'].fillna("", inplace=True)
print("After handling missing values")
print("Unique classes:", df_combined['label'].unique())

unique_classes = df_combined['label'].unique()
if len(unique_classes) < 2:
    raise ValueError("The dataset should contain at least two unique classes.")
X_train, X_test, y_train, y_test = train_test_split(df_combined['full_text'], df_combined['label'], test_size=0.2,random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

classifiers = {
    "SVM": SVC(kernel='linear'),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}
for name, clf in classifiers.items():
    print(f"Training {name} classifier...")
    clf.fit(X_train_tfidf, y_train)
    print(f"Evaluating {name} classifier...")
    y_pred = clf.predict(X_test_tfidf)
    print(f"Classifier: {name}")
    print(classification_report(y_test, y_pred))
    print("=" * 50)

    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

    joblib.dump(clf, f'{name.lower().replace(" ", "_")}_model.pkl')
