# Spam Email Classifier Using Machine Learning

## 1. Project Description
The Spam Email Classifier is a machine learning project that identifies whether a text message is spam or not spam. It uses Natural Language Processing (NLP) techniques to process text and a machine learning algorithm to classify messages.

## 2. Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- NLTK
- Streamlit
- TF-IDF Vectorization
- Multinomial Naive Bayes

## 3. Dataset
The project uses the SMS Spam Collection dataset containing 5,572 messages labelled as spam or ham.

## 4. Features
- Text preprocessing and stopword removal
- TF-IDF feature extraction
- Spam and non-spam message classification
- Model evaluation using accuracy, confusion matrix, and classification report
- Interactive web interface using Streamlit

## 5. Model Performance
The model achieved approximately 96.5% accuracy on the test dataset.

## 6. Project Structure
- `train_model.py` — Trains and evaluates the machine learning model.
- `app.py` — Runs the Streamlit web application.
- `spam_model.pkl` — Saved trained model.
- `vectorizer.pkl` — Saved TF-IDF vectorizer.
- `requirements.txt` — Required Python libraries.
- `data/SMSSpamCollection` — Dataset file.

## 7. How to Run the Project

### Install the required libraries
```bash
pip install -r requirements.txt
```

### Train the model
```bash
python train_model.py
```

### Start the web application
```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to use the application.

## 8. Conclusion
This project demonstrates how machine learning and NLP can be used to classify text messages as spam or not spam. The Streamlit interface makes predictions easy for users.
