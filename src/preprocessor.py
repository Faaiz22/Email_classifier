import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class EmailPreprocessor:
    """
    Advanced text preprocessing for email spam classification
    """
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
        self.label_encoder = LabelEncoder()
        
    def clean_text(self, text):
        """
        Comprehensive text cleaning
        """
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text):
        """
        Remove stopwords from text
        """
        tokens = word_tokenize(text)
        return ' '.join([word for word in tokens if word not in self.stop_words])
    
    def stem_text(self, text):
        """
        Apply stemming to text
        """
        tokens = word_tokenize(text)
        return ' '.join([self.stemmer.stem(word) for word in tokens])
    
    def lemmatize_text(self, text):
        """
        Apply lemmatization to text
        """
        tokens = word_tokenize(text)
        return ' '.join([self.lemmatizer.lemmatize(word) for word in tokens])
    
    def extract_features(self, text):
        """
        Extract additional features from text
        """
        features = {}
        
        # Length features
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        
        # Character features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['uppercase_count'] = sum(1 for c in text if c.isupper())
        features['digit_count'] = sum(1 for c in text if c.isdigit())
        
        # Spam indicator words
        spam_words = ['free', 'win', 'winner', 'cash', 'prize', 'urgent', 'limited', 'offer', 'click', 'now']
        features['spam_word_count'] = sum(1 for word in spam_words if word in text.lower())
        
        return features
    
    def preprocess_dataset(self, df, text_column='text', label_column='label', 
                          use_stemming=True, use_lemmatization=False, 
                          max_features=5000, use_tfidf=True):
        """
        Complete preprocessing pipeline
        """
        print("Starting preprocessing...")
        
        # Make a copy
        processed_df = df.copy()
        
        # Basic cleaning
        print("Cleaning text...")
        processed_df['cleaned_text'] = processed_df[text_column].apply(self.clean_text)
        
        # Remove stopwords
        print("Removing stopwords...")
        processed_df['no_stopwords'] = processed_df['cleaned_text'].apply(self.remove_stopwords)
        
        # Apply stemming or lemmatization
        if use_stemming:
            print("Applying stemming...")
            processed_df['processed_text'] = processed_df['no_stopwords'].apply(self.stem_text)
        elif use_lemmatization:
            print("Applying lemmatization...")
            processed_df['processed_text'] = processed_df['no_stopwords'].apply(self.lemmatize_text)
        else:
            processed_df['processed_text'] = processed_df['no_stopwords']
        
        # Extract additional features
        print("Extracting features...")
        feature_list = []
        for text in processed_df[text_column]:
            feature_list.append(self.extract_features(text))
        
        feature_df = pd.DataFrame(feature_list)
        processed_df = pd.concat([processed_df, feature_df], axis=1)
        
        # Vectorize text
        print("Vectorizing text...")
        if use_tfidf:
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
        
        # Fit and transform text
        text_features = self.vectorizer.fit_transform(processed_df['processed_text'])
        
        # Convert to DataFrame for easier handling
        feature_names = self.vectorizer.get_feature_names_out()
        text_feature_df = pd.DataFrame(text_features.toarray(), columns=feature_names)
        
        # Encode labels
        processed_df['label_encoded'] = self.label_encoder.fit_transform(processed_df[label_column])
        
        print(f"Preprocessing complete!")
        print(f"Original features: {len(processed_df.columns)}")
        print(f"Text features: {text_features.shape[1]}")
        print(f"Total samples: {len(processed_df)}")
        
        return processed_df, text_feature_df, text_features
    
    def preprocess_single_text(self, text):
        """
        Preprocess a single text for prediction
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Please run preprocess_dataset first.")
        
        # Clean text
        cleaned = self.clean_text(text)
        no_stopwords = self.remove_stopwords(cleaned)
        processed = self.stem_text(no_stopwords)
        
        # Vectorize
        vectorized = self.vectorizer.transform([processed])
        
        # Extract additional features
        features = self.extract_features(text)
        
        return vectorized, features

# Utility functions
def load_and_preprocess_data(filename, test_size=0.2):
    """
    Load and preprocess data with train-test split
    """
    df = pd.read_csv(filename)
    
    preprocessor = EmailPreprocessor()
    processed_df, text_features_df, text_features = preprocessor.preprocess_dataset(df)
    
    from sklearn.model_selection import train_test_split
    
    X = text_features
    y = processed_df['label_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, preprocessor, processed_df

if __name__ == "__main__":
    # Example usage
    from data_scraper import EmailDataScraper
    
    # Create sample dataset if not exists
    scraper = EmailDataScraper()
    scraper.scrape_sample_emails()
    df = scraper.create_dataset()
    
    # Preprocess
    preprocessor = EmailPreprocessor()
    processed_df, text_features_df, text_features = preprocessor.preprocess_dataset(df)
    
    print(f"\nPreprocessing Results:")
    print(f"Shape of processed text features: {text_features.shape}")
    print(f"Label distribution:")
    print(processed_df['label'].value_counts())