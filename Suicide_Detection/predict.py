import joblib
from .preprocess import preprocess_text # Corrected import statement

# Load the saved vectorizer and model
tfidf_vectorizer = joblib.load("Suicide_Detection/tfidf_vectorizer.joblib")
linear_svc_model = joblib.load("Suicide_Detection/linear_svc_model.joblib")

def predict_suicide(text):
    """
    Preprocesses input text, vectorizes it using the loaded TF-IDF vectorizer,
    and makes a prediction using the loaded LinearSVC model.

    Args:
        text (str): The input text to classify.

    Returns:
        int: The predicted class (0 for non-suicide, 1 for suicide).
    """
    # Preprocess the input text
    cleaned_text = preprocess_text(text)

    # Vectorize the preprocessed text
    vectorized_text = tfidf_vectorizer.transform([cleaned_text])

    # Make a prediction
    prediction = linear_svc_model.predict(vectorized_text)

    return prediction[0]

if __name__ == '__main__':
    # Example usage (optional, for testing)
    text1 = "I feel so sad and alone, I don't know what to do anymore."
    text2 = "Had a great day today, everything went well!"

    pred1 = predict_suicide(text1)
    pred2 = predict_suicide(text2)

    print(f"Prediction for text 1: {'suicide' if pred1 == 1 else 'non-suicide'}")
    print(f"Prediction for text 2: {'suicide' if pred2 == 1 else 'non-suicide'}")
