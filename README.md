#Mexico Sentiment Analysis using Machine Learning and TextBlob

This is a **Streamlit web application** that performs **sentiment analysis** on text related to **Mexico**, particularly sourced from its [Wikipedia page](https://en.wikipedia.org/wiki/Mexico). It uses both a **Logistic Regression model** trained on TF-IDF features and **TextBlob** to analyze sentiment polarity.

---

## 🚀 Features

- Input custom text related to Mexico
- Predict sentiment using:
  - ✅ Logistic Regression Model (TF-IDF)
  - ✅ TextBlob Polarity
- See model confidence and polarity scores
- View cleaned (preprocessed) version of your input
- Intuitive Streamlit UI

---

## 🧠 Technologies Used

- Python 🐍
- Scikit-learn
- TextBlob
- NLTK
- Pandas, NumPy
- Streamlit (for UI)
- BeautifulSoup (used for scraping Wikipedia in notebook)

---

## 📂 Folder Structure

Mexico_Sentiment_Analysis/ │ ├── Mexico_Sentiment_Analysis.py # Main Streamlit app ├── sentiment_model.pkl # Trained Logistic Regression model ├── vectorizer.pkl # TF-IDF Vectorizer ├── Mexico_L035.ipynb # Original Jupyter notebook (model training) └── README.md # You're here!


---

## 📥 How to Run

1. Clone or download the repo.
2. Make sure the following files are in the same folder:
   - `Mexico_Sentiment_Analysis.py`
   - `sentiment_model.pkl`
   - `vectorizer.pkl`

3. Install the required packages:
```bash
pip install -r requirements.txt

Launch the app:
streamlit run Mexico_Sentiment_Analysis.py

📋 Example
Input:

"Mexico has a rich culture and beautiful history."

Output:

ML Model: Positive (Confidence: 0.91)

TextBlob: Positive (Polarity: 0.5)
📚 Reference
Mexico Wikipedia Page

Applied Artificial Intelligence Project – SVKM's NMIMS Hyderabad

👨‍💻 Developed By
Narahari Abhinav

B.Tech CSE - Data Science

SVKM's NMIMS Hyderabad


Let me know if you want:
- A `requirements.txt` file
- A zipped folder with everything bundled
- Screenshots section added to the README  
Happy to help!
