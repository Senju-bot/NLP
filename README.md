#  **NLP Sentiment Classification System**

### *A Machine Learning & NLP Project Using Python, spaCy, NLTK, and Scikit-Learn*


---

##  **Overview**

This project implements a **Sentiment Classification System** using Natural Language Processing (NLP) and Machine Learning (ML). The system can classify any input sentence into:

✔ **Positive**
✔ **Negative**
✔ **Neutral**

It uses a combination of **spaCy (for lemmatization)**, **NLTK (for stopwords)**, and **Scikit-Learn (TF-IDF + Logistic Regression)**.

This project is ideal for:

* Students learning NLP
* Machine Learning beginners
* College projects and demos
* Simple sentiment analysis prototyping

---

#  **Key Features**

###  **1. NLP Text Preprocessing**

* Lowercasing
* Punctuation removal
* Stopword removal
* Tokenization
* Lemmatization using spaCy

###  **2. Machine Learning Pipeline**

* TF-IDF Vectorization
* Logistic Regression Classifier
* Auto-validation (train/test split)

###  **3. Real-Time User Prediction**

You can input text directly into the terminal and get instant sentiment results.

###  **4. Clean, Modular Codebase**

Easy to expand with more features such as:

* NER
* Deep learning models (LSTM, BERT)
* Larger datasets

---

#  **System Architecture Diagram**

```
                         ┌────────────────────────────────┐
                         │        User Input Text          │
                         └────────────────────────────────┘
                                        │
                                        ▼
                         ┌────────────────────────────────┐
                         │      Preprocessing Layer        │
                         │  - Lowercasing                  │
                         │  - Cleaning (regex)             │
                         │  - Stopword Removal (NLTK)      │
                         │  - Lemmatization (spaCy)        │
                         └────────────────────────────────┘
                                        │
                                        ▼
                         ┌────────────────────────────────┐
                         │      Feature Engineering        │
                         │     TF-IDF Vectorization        │
                         └────────────────────────────────┘
                                        │
                                        ▼
                         ┌────────────────────────────────┐
                         │  Logistic Regression Classifier │
                         │  (Scikit-Learn Model Training)  │
                         └────────────────────────────────┘
                                        │
                                        ▼
                         ┌────────────────────────────────┐
                         │     Sentiment Prediction        │
                         │ Positive / Negative / Neutral   │
                         └────────────────────────────────┘
```

---

#  **Repository Structure**

```
NLP-Sentiment-Classifier/
│
├── nlp_program.py            # Main script
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

---

#  **Technologies & Libraries**

| Technology         | Purpose                     |
| ------------------ | --------------------------- |
| **Python**         | Core programming language   |
| **spaCy**          | Lemmatization, NLP pipeline |
| **NLTK**           | Stopwords                   |
| **Scikit-Learn**   | Vectorization + ML model    |
| **Regex / string** | Text cleaning               |
| **pandas**         | Dataset handling            |

---

#  **Installation Guide**

###  1. Clone the Repository

```
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

###  2. Install Dependencies

```
pip install -r requirements.txt
```

###  3. Download spaCy Model

```
python -m spacy download en_core_web_sm
```

###  4. (Optional) Download NLTK Stopwords

```python
import nltk
nltk.download("stopwords")
```

---

#  **How to Run the Program**

Run the main script:

```
python nlp_program.py
```

You will be prompted with:

```
Enter a sentence to analyze sentiment:
```

Example:

```
Enter a sentence to analyze sentiment: I absolutely love this!
Predicted Sentiment: positive
```

---

#  **Model Performance**

The script automatically prints a classification report using the test split:

```
Model Evaluation:
              precision    recall  f1-score   support
...
```

This includes:

* Precision
* Recall
* F1 Score
* Support

---

#  **Dataset Used (Built-in Mini Dataset)**

| Example Text                           | Label    |
| -------------------------------------- | -------- |
| I love this product, it's amazing!     | positive |
| This is the worst experience ever.     | negative |
| The service was okay, nothing special. | neutral  |
| Absolutely fantastic!                  | positive |
| Terrible. I will never buy this again. | negative |
