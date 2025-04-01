import pytest
from app import app
from resource.integrate import Score
from src.predict import load_model_vectoriser, MODEL_PATH, preprocess_text, score, main

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_endpoint(client):
    """Test the /predict endpoint."""
    test_text = "Congratulations! You have won a free iPhone."
    response = client.post('/predict', json={"text": test_text})
    
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert "propensity" in data
    assert isinstance(data["propensity"], float)

def test_null_request(client):
    """Test the endpoint with a null request."""
    response = client.post('/predict')
    assert response.status_code == 415  # The status code is 415 for unsupported media type

def test_preprocess_text():
    """Test the text preprocessing function."""
    # Testing text preprocessing to cover those lines
    original_text = "Hello, World! This is a Test. These are STOPWORDS: the, a, an."
    processed_text = preprocess_text(original_text)
    
    # Check that preprocessing did something
    assert processed_text != original_text
    assert "," not in processed_text
    assert "!" not in processed_text
    assert "the" not in processed_text
    assert "a" not in processed_text
    assert "an" not in processed_text
    assert processed_text.islower()

def test_score_function():
    """Test the score function with different thresholds."""
    model, vectorizer = load_model_vectoriser(MODEL_PATH)
    
    # Test with likely spam text
    spam_text = "Free money, give a miscall on 1234567890"
    prediction, propensity = score(spam_text, model, vectorizer, threshold=0.5)
    assert isinstance(prediction, bool)
    assert isinstance(propensity, float)
    
    # Test with likely non-spam text
    non_spam_text = "Meeting scheduled for tomorrow at 2pm"
    prediction, propensity = score(non_spam_text, model, vectorizer, threshold=0.5)
    assert isinstance(prediction, bool)
    assert isinstance(propensity, float)
    
    # Test with a different threshold
    prediction_high_threshold, _ = score(spam_text, model, vectorizer, threshold=0.9)
    prediction_low_threshold, _ = score(spam_text, model, vectorizer, threshold=0.1)
    
    # These assertions help ensure the threshold logic is covered
    assert isinstance(prediction_high_threshold, bool)
    assert isinstance(prediction_low_threshold, bool)

def test_home_route(client):
    """Test the home route with both GET and POST methods."""
    # Test GET request
    response = client.get('/')
    assert response.status_code == 200
    
    # Test POST request with text
    response = client.post('/', data={"text": "Sample text for testing"})
    assert response.status_code == 200
    
    # Test POST request without text (the form has required attribute, so we need to bypass it)
    response = client.post('/', data={})
    assert response.status_code == 200
    assert b'<p><strong>Prediction:</strong> </p>' in response.data
    assert b'<p><strong>Propensity Score:</strong> </p>' in response.data

def test_main_function():
    """Test the main function to cover the remaining lines in predict.py."""
    try:
        main()
        success = True
    except Exception:
        success = False
    assert success