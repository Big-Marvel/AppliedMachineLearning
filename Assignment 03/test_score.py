import pytest
from score import load_model_vectoriser, score

# Fixture to load model and vectorizer
@pytest.fixture(scope="module")
def model_and_vectoriser():
    model_path = r"C:\Users\Keshav\Desktop\DS Course\Sem 4\3. AML\Assignment 3\Model"
    return load_model_vectoriser(model_path)

# Sample texts for testing
@pytest.fixture
def sample_texts():
    return {
        "spam": "Congratulations! You have won a $1000.",
        "non_spam": "Hello, how are you doing today?"
    }



def test_smoke_test(model_and_vectoriser, sample_texts):
    """
    Test to ensure the function runs without crashing.
    """
    model, vectoriser = model_and_vectoriser
    prediction, propensity = score(sample_texts["spam"], model, vectoriser, threshold=0.5)
    assert isinstance(prediction, bool), "Prediction should be a boolean"
    assert isinstance(propensity, float), "Propensity should be a float"



def test_format_test(model_and_vectoriser, sample_texts):
    """
    Test to ensure input/output formats/types are as expected.
    """
    model, vectoriser = model_and_vectoriser
    prediction, propensity = score(sample_texts["spam"], model, vectoriser, threshold=0.5)
    
    assert isinstance(prediction, bool), "Prediction should be a boolean"
    assert isinstance(propensity, float), "Propensity should be a float"



def test_prediction_value(model_and_vectoriser, sample_texts):
    """
    Test to check if the prediction value is 0 or 1.
    """
    model, vectoriser = model_and_vectoriser
    prediction, _ = score(sample_texts["spam"], model, vectoriser, threshold=0.5)
    
    # Convert boolean to integer (True -> 1, False -> 0)
    assert int(prediction) in [0, 1], "Prediction value should be either 0 or 1"



def test_propensity_range(model_and_vectoriser, sample_texts):
    """
    Test to ensure the propensity score is between 0 and 1.
    """
    model, vectoriser = model_and_vectoriser
    _, propensity = score(sample_texts["spam"], model, vectoriser, threshold=0.5)
    
    assert 0.0 <= propensity <= 1.0, "Propensity score should be between 0 and 1"



def test_threshold_zero(model_and_vectoriser, sample_texts):
    """
    Test to ensure that with threshold set to 0 the prediction is always True (1).
    """
    model, vectoriser = model_and_vectoriser
    prediction, _ = score(sample_texts["spam"], model, vectoriser, threshold=0.0)
    
    assert prediction is True, "With threshold 0, prediction should always be True"



def test_threshold_one(model_and_vectoriser, sample_texts):
    """
    Test to ensure that with threshold set to 1 the prediction is always False (0).
    """
    model, vectorizer = model_and_vectoriser
    prediction, _ = score(sample_texts["non_spam"], model, vectorizer, threshold=1.0)
    
    assert prediction is False, "With threshold 1, prediction should always be False"



def test_spam_detection(model_and_vectoriser):
    """
    Test to check if an obvious spam input text gets classified as spam (prediction is True).
    """
    model, vectorizer = model_and_vectoriser
    spam_text = "Congratulations! You've won a  gift card! Click here to claim your prize"
    
    prediction, _ = score(spam_text, model, vectorizer, threshold=0.5)
    
    assert prediction is True, "Obvious spam text should be classified as spam"


def test_non_spam_detection(model_and_vectoriser):
    """
    Test to check if an obvious non-spam input text gets classified as non-spam (prediction is False).
    """
    model, vectorizer = model_and_vectoriser
    non_spam_text = "Hello there! How's your day going?"
    
    prediction, _ = score(non_spam_text, model, vectorizer, threshold=0.5)
    
    assert prediction is False, "Obvious non-spam text should be classified as non-spam"

if __name__ == "__main__":
    pytest.main()
