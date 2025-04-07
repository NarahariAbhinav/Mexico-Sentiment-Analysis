import streamlit as st

# Set page configuration first - this must be the first Streamlit command
st.set_page_config(
    page_title="Mexico Sentiment Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pickle
import nltk
from textblob import TextBlob
import string
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.linear_model import LogisticRegression
import numpy as np

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load trained model and TF-IDF vectorizer with fallback
@st.cache_resource
def load_models():
    try:
        model = pickle.load(open("sentiment_model.pkl", "rb"))
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
        st.sidebar.success("✅ Loaded trained model successfully")
    except FileNotFoundError:
        st.sidebar.warning("⚠️ Model files not found. Using backup TextBlob-based model.")
        # Create a simple vectorizer as fallback
        vectorizer = TfidfVectorizer(max_features=5000)
        # We'll use this as a placeholder - actual predictions will use TextBlob
        model = LogisticRegression()
    return model, vectorizer

# Check if the model file exists
model_exists = os.path.exists("sentiment_model.pkl") and os.path.exists("vectorizer.pkl")
model, vectorizer = load_models()

# Function to preprocess input text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[[0-9]*\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\W', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

# Function to get textblob-based prediction (as backup when model is missing)
def textblob_prediction(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    # Convert TextBlob polarity to binary sentiment and probability
    if polarity > 0:
        # Convert polarity to probability-like score between 0.5 and 1.0
        proba = 0.5 + (polarity / 2)
        return 1, proba  # Positive
    else:
        # Convert negative polarity to probability-like score between 0.5 and 1.0
        proba = 0.5 + (abs(polarity) / 2)
        return 0, proba  # Negative

# Function to get model prediction with fallback to TextBlob
def get_prediction(text, vector_input):
    if model_exists:
        # Use the trained model
        prediction = model.predict(vector_input)[0]
        proba = model.predict_proba(vector_input).max()
    else:
        # Use TextBlob as fallback
        prediction, proba = textblob_prediction(text)
    
    return prediction, proba

# Function to get detailed sentiment analysis
def detailed_sentiment_analysis(text):
    blob = TextBlob(text)
    sentences = blob.sentences
    
    sentence_analysis = []
    for sentence in sentences:
        polarity = sentence.sentiment.polarity
        subjectivity = sentence.sentiment.subjectivity
        
        if polarity > 0.1:
            sentiment = "Positive"
        elif polarity < -0.1:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
            
        sentence_analysis.append({
            "sentence": str(sentence),
            "polarity": polarity,
            "subjectivity": subjectivity,
            "sentiment": sentiment
        })
    
    return pd.DataFrame(sentence_analysis)

def generate_wordcloud(text):
    try:
        stop_words = set(stopwords.words('english'))
        wordcloud = WordCloud(width=800, height=400, 
                            background_color='white', 
                            stopwords=stop_words,
                            max_words=100).generate(text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        return plt
    except:
        # Fallback if WordCloud fails
        st.warning("Could not generate word cloud - make sure wordcloud package is installed")
        return None

# Add a sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fc/Flag_of_Mexico.svg", width=100)
    st.title("About")
    st.info(
        """
        This app analyzes sentiment in text using both a custom ML model trained on 
        Mexico-related content and TextBlob's general sentiment analysis.
        
        The custom model was trained on text scraped from the 
        [Mexico Wikipedia page](https://en.wikipedia.org/wiki/Mexico).
        """
    )
    
    if not model_exists:
        st.warning("""
        ⚠️ Model files not found. Using TextBlob for all predictions.
        
        To use the custom model, ensure these files exist:
        - sentiment_model.pkl
        - vectorizer.pkl
        """)
    
    st.subheader("Features")
    st.markdown("""
    - Sentiment prediction with confidence scores
    - Sentence-by-sentence analysis
    - Word cloud visualization
    - Comparison of different text sentiments
    """)
    
    st.markdown("---")
    st.caption("Developed by Narahari Abhinav | Applied AI Project | NMIMS Hyderabad")

# Main page content
st.title("🇲🇽 Mexico Sentiment Analysis")
st.markdown("Analyze how text relates to Mexico using advanced sentiment analysis techniques.")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Analysis", "Advanced Features", "Documentation"])

with tab1:
    # Input text
    user_input = st.text_area("Enter any sentence or paragraph to analyze:", height=150)
    
    col1, col2 = st.columns(2)
    with col1:
        analyze_button = st.button("Analyze Sentiment", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("Clear", type="secondary", use_container_width=True)
        if clear_button:
            st.session_state.user_input = ""
            st.experimental_rerun()
    
    if analyze_button:
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text.")
        else:
            # Create columns for results
            result_col1, result_col2 = st.columns(2)
            
            # Preprocess and vectorize input
            cleaned = preprocess_text(user_input)
            
            try:
                vector_input = vectorizer.transform([cleaned])
                
                # Get prediction (with fallback mechanism)
                prediction, proba = get_prediction(user_input, vector_input)
                sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
                
                with result_col1:
                    model_name = "ML Model" if model_exists else "TextBlob Model"
                    st.subheader(f"🔍 {model_name} Prediction")
                    st.markdown(f"<h3 style='text-align: center;'>Sentiment: {sentiment}</h3>", unsafe_allow_html=True)
                    
                    # Create a progress bar for confidence
                    st.markdown(f"<p style='text-align: center;'>Confidence: {proba:.2f}</p>", unsafe_allow_html=True)
                    st.progress(float(proba))
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
            
            # TextBlob sentiment - this always works
            try:
                blob = TextBlob(user_input)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                blob_sentiment = "Positive 😊" if polarity > 0.1 else "Negative 😞" if polarity < -0.1 else "Neutral 😐"
                
                with result_col2:
                    st.subheader("🧠 TextBlob Analysis")
                    st.markdown(f"<h3 style='text-align: center;'>Sentiment: {blob_sentiment}</h3>", unsafe_allow_html=True)
                    
                    # Create a progress bar for polarity
                    normalized_polarity = (polarity + 1) / 2  # Convert from [-1,1] to [0,1]
                    st.markdown(f"<p style='text-align: center;'>Polarity: {polarity:.2f}</p>", unsafe_allow_html=True)
                    st.progress(float(normalized_polarity))
                    
                    st.markdown(f"<p style='text-align: center;'>Subjectivity: {subjectivity:.2f}</p>", unsafe_allow_html=True)
                    st.progress(float(subjectivity))
            except Exception as e:
                st.error(f"Error in TextBlob analysis: {str(e)}")
            
            # Detailed analysis section
            st.subheader("📊 Detailed Analysis")
            
            # WordCloud
            st.subheader("Word Cloud")
            try:
                word_cloud_plt = generate_wordcloud(user_input)
                if word_cloud_plt:
                    st.pyplot(word_cloud_plt)
            except Exception as e:
                st.error(f"Error generating word cloud: {str(e)}")
                st.warning("Make sure to install WordCloud: pip install wordcloud")
            
            # Per-sentence analysis
            try:
                blob = TextBlob(user_input)
                if len(blob.sentences) > 1:
                    st.subheader("Sentence-by-Sentence Analysis")
                    df_sentences = detailed_sentiment_analysis(user_input)
                    
                    # Display the dataframe
                    st.dataframe(df_sentences, use_container_width=True)
                    
                    # Plot sentence polarities
                    st.subheader("Sentence Polarity Distribution")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    
                    # Create bar chart
                    colors = ['#ff9999' if x < -0.1 else '#99ff99' if x > 0.1 else '#9999ff' for x in df_sentences['polarity']]
                    sns.barplot(x=df_sentences.index, y='polarity', data=df_sentences, palette=colors, ax=ax)
                    
                    ax.set_title('Sentiment Polarity by Sentence')
                    ax.set_xlabel('Sentence Number')
                    ax.set_ylabel('Polarity (-1 to +1)')
                    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Error in sentence analysis: {str(e)}")
            
            # Optional text preview
            with st.expander("🔧 Show Preprocessed Text"):
                st.code(cleaned)

with tab2:
    st.header("Compare Texts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Text 1")
        text1 = st.text_area("Enter first text:", height=150, key="text1")
    
    with col2:
        st.subheader("Text 2")
        text2 = st.text_area("Enter second text:", height=150, key="text2")
    
    if st.button("Compare Sentiments", type="primary"):
        if text1.strip() == "" or text2.strip() == "":
            st.warning("⚠️ Please enter both texts to compare.")
        else:
            try:
                # Process text 1
                cleaned1 = preprocess_text(text1)
                vector_input1 = vectorizer.transform([cleaned1])
                prediction1, proba1 = get_prediction(text1, vector_input1)
                blob1 = TextBlob(text1)
                polarity1 = blob1.sentiment.polarity
                
                # Process text 2
                cleaned2 = preprocess_text(text2)
                vector_input2 = vectorizer.transform([cleaned2])
                prediction2, proba2 = get_prediction(text2, vector_input2)
                blob2 = TextBlob(text2)
                polarity2 = blob2.sentiment.polarity
                
                # Create comparison dataframe
                model_name = "ML Model" if model_exists else "Backup Model"
                comparison_data = {
                    'Metric': [f'{model_name} Prediction', f'{model_name} Confidence', 'TextBlob Polarity', 'TextBlob Sentiment'],
                    'Text 1': [
                        "Positive" if prediction1 == 1 else "Negative",
                        f"{proba1:.2f}",
                        f"{polarity1:.2f}",
                        "Positive" if polarity1 > 0.1 else "Negative" if polarity1 < -0.1 else "Neutral"
                    ],
                    'Text 2': [
                        "Positive" if prediction2 == 1 else "Negative",
                        f"{proba2:.2f}",
                        f"{polarity2:.2f}",
                        "Positive" if polarity2 > 0.1 else "Negative" if polarity2 < -0.1 else "Neutral"
                    ]
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                st.table(comparison_df)
                
                # Visualize comparison
                st.subheader("Visual Comparison")
                
                # Create bar chart for comparison
                fig, ax = plt.subplots(figsize=(10, 6))
                
                bar_data = pd.DataFrame({
                    'Text': ['Text 1', 'Text 2'],
                    f'{model_name} Confidence': [proba1, proba2],
                    'TextBlob Polarity': [(polarity1 + 1)/2, (polarity2 + 1)/2]  # Normalize to [0,1]
                })
                
                bar_data_melted = pd.melt(bar_data, id_vars=['Text'], value_vars=[f'{model_name} Confidence', 'TextBlob Polarity'])
                
                sns.barplot(x='Text', y='value', hue='variable', data=bar_data_melted, ax=ax)
                ax.set_title('Sentiment Comparison')
                ax.set_ylabel('Score (0-1)')
                plt.tight_layout()
                
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error in comparison: {str(e)}")
                st.warning("Make sure all required packages are installed")

with tab3:
    st.header("Documentation")
    
    st.subheader("How It Works")
    st.write("""
    This app uses two different approaches to sentiment analysis:
    
    1. **Mexico-specific Model**: Designed to analyze text sentiment related to Mexico.
       - Uses TF-IDF vectorization to convert text into features
       - Predicts binary sentiment (Positive/Negative)
       - Provides confidence scores
       - Falls back to TextBlob if model files are missing
    
    2. **TextBlob Sentiment Analysis**: A general-purpose sentiment analyzer.
       - Provides polarity scores from -1 (negative) to +1 (positive)
       - Provides subjectivity scores from 0 (objective) to 1 (subjective)
    """)
    
    st.subheader("Required Packages")
    st.code("""
    pip install streamlit textblob nltk scikit-learn pandas matplotlib seaborn wordcloud
    """)
    
    st.subheader("Interpreting Results")
    st.write("""
    - **ML Model Prediction**: Shows whether the text is likely to express positive or negative sentiment about Mexico.
    - **Confidence**: How certain the model is about its prediction.
    - **Polarity**: TextBlob's measure of positive/negative sentiment (-1 to +1).
    - **Subjectivity**: How subjective or opinionated the text is (0 to 1).
    - **Word Cloud**: Visualizes the most common words in your text.
    - **Sentence Analysis**: Breaks down longer texts to analyze sentiment sentence by sentence.
    """)
    
    st.subheader("Tips for Use")
    st.write("""
    - For best results, enter text specifically related to Mexico
    - Compare different analyses to get a more nuanced understanding
    - Use the sentence-by-sentence breakdown for longer texts
    - The compare feature is useful for A/B testing different phrasings
    """)
    
    st.subheader("Getting Started with Model Training")
    with st.expander("How to create your own model"):
        st.markdown("""
        If you want to create the model files yourself, here's a simple example:
        
        ```python
        import pandas as pd
        import pickle
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        
        # Load your dataset (example format)
        # df = pd.DataFrame({'text': ['...'], 'sentiment': [0, 1, ...]})
        
        # Create vectorizer
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df['text'])
        y = df['sentiment']
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        # Save model and vectorizer
        pickle.dump(model, open("sentiment_model.pkl", "wb"))
        pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
        ```
        """)