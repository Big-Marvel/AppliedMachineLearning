## Problem 

## 1. containerization

- create a docker container for the flask app created in Assignment 3
- create a Dockerfile which contains the instructions to build the container, which included
	- installing the dependencies
	- copying app.py and score.py
	- launching the app by running “python app.py” upon entry
- build the docker image using Dockerfile
- run the docker container with appropriate port bindings
- in test.py write test_docker(..) function which does the following
	- launches the docker container using commandline (e.g. os.sys(..), 	docker build and docker run)
	- sends a request to the localhost endpoint /score (e.g. using 		requests library) for a sample text
	- checks if the response is as expected
	- close the docker container

In coverage.txt, produce the coverage report using pytest for the tests in test.py

## 2. continuous integration
- write a pre-commit git hook that will run the test.py automatically every time you try to commit the code to your local ‘main’ branch
- copy and push this pre-commit git hook file to your git repo

---
## Solution

### 1. Frontend:
The user interacts with the web app through the index.html template.
They input text into a form and submit it for spam prediction.

### 2. Backend:
The Flask app receives the input text and sends it to the /predict API endpoint.
The /predict endpoint preprocesses the text, generates a prediction, and returns the result.

### 3. Machine Learning:
The score.py and predict.py modules handle text preprocessing and prediction using the ML model and vectorizer.

### 4. Containerization:
The Dockerfile defines how to containerize the app, ensuring it runs consistently across environments.

### 5. Testing:
The test.py script verifies that the Docker container builds and runs correctly and that the API endpoints function as expected.
