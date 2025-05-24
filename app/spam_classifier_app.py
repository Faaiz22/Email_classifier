import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import warnings
warnings.filterwarnings('ignore')

# Try importing sklearn components with error handling
try:
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
    from sklearn.utils.class_weight import compute_class_weight
    SKLEARN_AVAILABLE = True
except ImportError as e:
    st.error(f"Scikit-learn import error: {e}")
    SKLEARN_AVAILABLE = False

# Try importing visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Try importing NLTK with error handling
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Download NLTK data with error handling
@st.cache_resource
def download_nltk_data():
    if not NLTK_AVAILABLE:
        return []
    try:
        nltk.download('stopwords', quiet=True)
        return stopwords.words('english')
    except Exception as e:
        st.warning(f"Could not download NLTK data: {e}")
        return []

# Check if required libraries are available
def check_dependencies():
    missing_deps = []
    if not SKLEARN_AVAILABLE:
        missing_deps.append("scikit-learn")
    if not MATPLOTLIB_AVAILABLE:
        missing_deps.append("matplotlib")
    if not NLTK_AVAILABLE:
        missing_deps.append("nltk")
    
    return missing_deps

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'performance_metrics' not in st.session_state:
    st.session_state.performance_metrics = {}

class SpamClassifier:
    def __init__(self):
        self.stop_words = download_nltk_data() if NLTK_AVAILABLE else []
        self.stemmer = PorterStemmer() if NLTK_AVAILABLE else None
        
    def preprocess_text(self, text):
        """
        Comprehensive text preprocessing pipeline
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove stopwords and stem (if NLTK is available)
        if self.stop_words and self.stemmer:
            words = [self.stemmer.stem(word) for word in text.split() 
                    if word not in self.stop_words and len(word) > 2]
        else:
            # Basic word filtering if NLTK is not available
            common_stopwords = ['the', 'is', 'at', 'which', 'on', 'and', 'a', 'to', 'for', 'of', 'with', 'in']
            words = [word for word in text.split() if word not in common_stopwords and len(word) > 2]
        
        return ' '.join(words)
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """
        Train multiple models with class balancing
        """
        # Calculate class weights for imbalanced dataset
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        models = {}
        performance = {}
        
        # Logistic Regression with balanced class weights
        lr_model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000,
            solver='liblinear'
        )
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        
        models['Logistic Regression'] = lr_model
        performance['Logistic Regression'] = self.calculate_metrics(y_test, lr_pred)
        
        # Naive Bayes with smoothing
        nb_model = MultinomialNB(alpha=1.0)
        nb_model.fit(X_train, y_train)
        nb_pred = nb_model.predict(X_test)
        
        models['Naive Bayes'] = nb_model
        performance['Naive Bayes'] = self.calculate_metrics(y_test, nb_pred)
        
        return models, performance
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate comprehensive performance metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, pos_label='spam', average='binary'),
            'recall': recall_score(y_true, y_pred, pos_label='spam', average='binary'),
            'f1_score': f1_score(y_true, y_pred, pos_label='spam', average='binary'),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred)
        }

def main():
    st.set_page_config(
        page_title="🚫 Spam Email Classifier",
        page_icon="🚫",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Check dependencies first
    missing_deps = check_dependencies()
    if missing_deps:
        st.error(f"⚠️ Missing required dependencies: {', '.join(missing_deps)}")
        st.info("Please install the missing packages using: pip install " + " ".join(missing_deps))
        st.code(f"pip install {' '.join(missing_deps)}")
        return
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B6B;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF6B6B;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🚫 Spam Email Classifier</h1>', unsafe_allow_html=True)
    
    classifier = SpamClassifier()
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox("Choose a page:", 
                               ["🏠 Home", "📤 Upload & Train", "🔍 Classify Text", "📈 Model Performance", "ℹ️ About"])
    
    if page == "🏠 Home":
        home_page()
    elif page == "📤 Upload & Train":
        upload_and_train_page(classifier)
    elif page == "🔍 Classify Text":
        classify_text_page()
    elif page == "📈 Model Performance":
        performance_page()
    elif page == "ℹ️ About":
        about_page()

def home_page():
    st.markdown("""
    ## Welcome to the Spam Email Classifier! 🎯
    
    This application uses advanced machine learning techniques to classify emails as **spam** or **ham** (legitimate).
    
    ### 🚀 Features:
    - **Multiple ML Models**: Logistic Regression & Naive Bayes
    - **Advanced Preprocessing**: Text cleaning, stemming, and feature extraction
    - **Class Balancing**: Handles imbalanced datasets effectively
    - **Real-time Classification**: Test your own text
    - **Performance Metrics**: Comprehensive model evaluation
    
    ### 📋 How to Use:
    1. **Upload & Train**: Upload your CSV file and train the models
    2. **Classify Text**: Test the classifier with your own text
    3. **View Performance**: Analyze model metrics and visualizations
    
    ### 📊 Expected CSV Format:
    Your CSV file should have columns named **'label'** and **'message'** (or similar):
    ```
    label,message
    ham,"Hello, how are you today?"
    spam,"URGENT! Claim your prize now!"
    ```
    """)
    
    # Quick stats if model is trained
    if st.session_state.model_trained:
        st.success("✅ Models are trained and ready for classification!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Models Available", len(st.session_state.models))
        with col2:
            best_model = max(st.session_state.performance_metrics.items(), 
                           key=lambda x: x[1]['f1_score'])
            st.metric("Best Model", best_model[0])
        with col3:
            st.metric("Best F1-Score", f"{best_model[1]['f1_score']:.3f}")

def upload_and_train_page(classifier):
    st.header("📤 Upload Dataset & Train Models")
    
    # Check if sklearn is available
    if not SKLEARN_AVAILABLE:
        st.error("⚠️ Scikit-learn is not available. Please install it to use this feature.")
        st.code("pip install scikit-learn")
        return
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
            
            # Display first few rows
            st.subheader("📋 Data Preview")
            st.dataframe(df.head())
            
            # Column selection
            st.subheader("🔧 Configure Columns")
            col1, col2 = st.columns(2)
            
            with col1:
                label_col = st.selectbox("Select Label Column:", df.columns)
            with col2:
                text_col = st.selectbox("Select Text Column:", df.columns)
            
            if st.button("🚀 Start Training", type="primary"):
                with st.spinner("Training models... This may take a few minutes."):
                    
                    # Data exploration
                    st.subheader("📊 Dataset Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Samples", len(df))
                    with col2:
                        spam_count = len(df[df[label_col] == 'spam'])
                        st.metric("Spam Messages", spam_count)
                    with col3:
                        ham_count = len(df[df[label_col] == 'ham'])
                        st.metric("Ham Messages", ham_count)
                    
                    # Class distribution
                    fig, ax = plt.subplots(figsize=(8, 5))
                    df[label_col].value_counts().plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'])
                    ax.set_title('Class Distribution')
                    ax.set_xlabel('Label')
                    ax.set_ylabel('Count')
                    plt.xticks(rotation=0)
                    st.pyplot(fig)
                    
                    # Preprocessing
                    st.info("🔄 Preprocessing text data...")
                    df['processed_text'] = df[text_col].apply(classifier.preprocess_text)
                    
                    # Feature extraction
                    st.info("🔍 Extracting features...")
                    vectorizer = TfidfVectorizer(
                        max_features=5000,
                        min_df=2,
                        max_df=0.95,
                        ngram_range=(1, 2)
                    )
                    
                    X = vectorizer.fit_transform(df['processed_text'])
                    y = df[label_col]
                    
                    # Train-test split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    # Train models
                    st.info("🤖 Training machine learning models...")
                    models, performance = classifier.train_models(X_train, X_test, y_train, y_test)
                    
                    # Store in session state
                    st.session_state.models = models
                    st.session_state.vectorizer = vectorizer
                    st.session_state.performance_metrics = performance
                    st.session_state.model_trained = True
                    
                    st.success("🎉 Models trained successfully!")
                    
                    # Quick performance overview
                    st.subheader("⚡ Quick Performance Overview")
                    perf_df = pd.DataFrame(performance).T
                    perf_df = perf_df[['accuracy', 'precision', 'recall', 'f1_score']].round(4)
                    st.dataframe(perf_df)
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV has the correct format with 'label' and 'message' columns.")

def classify_text_page():
    st.header("🔍 Classify Your Text")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the models first in the 'Upload & Train' section.")
        return
    
    # Text input methods
    input_method = st.radio("Choose input method:", ["✍️ Type Text", "📁 Upload Text File"])
    
    text_to_classify = ""
    
    if input_method == "✍️ Type Text":
        text_to_classify = st.text_area(
            "Enter text to classify:",
            placeholder="Type or paste your email content here...",
            height=150
        )
    else:
        uploaded_text_file = st.file_uploader("Upload a text file", type=['txt'])
        if uploaded_text_file:
            text_to_classify = str(uploaded_text_file.read(), "utf-8")
            st.text_area("File content:", text_to_classify, height=150, disabled=True)
    
    if text_to_classify and st.button("🎯 Classify Text", type="primary"):
        
        # Preprocess the text
        classifier = SpamClassifier()
        processed_text = classifier.preprocess_text(text_to_classify)
        
        # Transform using the trained vectorizer
        text_features = st.session_state.vectorizer.transform([processed_text])
        
        st.subheader("📊 Classification Results")
        
        results = []
        for model_name, model in st.session_state.models.items():
            prediction = model.predict(text_features)[0]
            probability = model.predict_proba(text_features)[0]
            
            results.append({
                'Model': model_name,
                'Prediction': prediction,
                'Confidence': max(probability),
                'Spam Probability': probability[1] if len(probability) > 1 else (1 - probability[0]),
                'Ham Probability': probability[0] if len(probability) > 1 else probability[0]
            })
        
        # Display results
        for result in results:
            col1, col2 = st.columns([2, 3])
            
            with col1:
                if result['Prediction'] == 'spam':
                    st.markdown(f"""
                    <div class="warning-card">
                        <h4>🚨 {result['Model']}</h4>
                        <h3>SPAM DETECTED</h3>
                        <p>Confidence: {result['Confidence']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="success-card">
                        <h4>✅ {result['Model']}</h4>
                        <h3>HAM (Legitimate)</h3>
                        <p>Confidence: {result['Confidence']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                # Probability chart
                fig, ax = plt.subplots(figsize=(6, 3))
                categories = ['Ham', 'Spam']
                probabilities = [result['Ham Probability'], result['Spam Probability']]
                colors = ['#4ECDC4', '#FF6B6B']
                
                bars = ax.bar(categories, probabilities, color=colors, alpha=0.7)
                ax.set_ylim(0, 1)
                ax.set_ylabel('Probability')
                ax.set_title(f'{result["Model"]} - Prediction Probabilities')
                
                # Add value labels on bars
                for bar, prob in zip(bars, probabilities):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{prob:.1%}', ha='center', va='bottom')
                
                st.pyplot(fig)
        
        # Consensus prediction
        spam_votes = sum(1 for result in results if result['Prediction'] == 'spam')
        consensus = 'spam' if spam_votes > len(results) / 2 else 'ham'
        
        st.subheader("🏆 Consensus Prediction")
        if consensus == 'spam':
            st.error(f"🚨 **SPAM DETECTED** (Voted by {spam_votes}/{len(results)} models)")
        else:
            st.success(f"✅ **HAM - Legitimate Email** (Voted by {len(results) - spam_votes}/{len(results)} models)")

def performance_page():
    st.header("📈 Model Performance Analysis")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the models first in the 'Upload & Train' section.")
        return
    
    # Performance metrics table
    st.subheader("📊 Performance Metrics Comparison")
    perf_df = pd.DataFrame(st.session_state.performance_metrics).T
    metrics_df = perf_df[['accuracy', 'precision', 'recall', 'f1_score']].round(4)
    st.dataframe(metrics_df, use_container_width=True)
    
    # Metrics visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        values = [st.session_state.performance_metrics[model][metric] for model in st.session_state.models.keys()]
        bars = ax.bar(st.session_state.models.keys(), values, color=['#FF6B6B', '#4ECDC4'])
        ax.set_title(f'{metric.title()} Comparison')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Confusion matrices
    st.subheader("🔍 Confusion Matrices")
    
    col1, col2 = st.columns(2)
    for i, (model_name, metrics) in enumerate(st.session_state.performance_metrics.items()):
        with col1 if i == 0 else col2:
            fig, ax = plt.subplots(figsize=(6, 5))
            cm = metrics['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
            ax.set_title(f'{model_name} - Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
    
    # Detailed classification reports
    st.subheader("📋 Detailed Classification Reports")
    for model_name, metrics in st.session_state.performance_metrics.items():
        with st.expander(f"📊 {model_name} Classification Report"):
            st.text(metrics['classification_report'])

def about_page():
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ## 🎯 Project Overview
    
    This **Spam Email Classifier** is a comprehensive machine learning application that demonstrates:
    
    ### 🔧 Technical Features:
    - **Advanced Text Preprocessing**: URL removal, email cleaning, stemming, stopword removal
    - **Feature Engineering**: TF-IDF vectorization with n-grams
    - **Multiple ML Models**: Logistic Regression and Naive Bayes
    - **Class Imbalance Handling**: Balanced class weights and stratified sampling
    - **Comprehensive Evaluation**: Multiple metrics and visualizations
    
    ### 📊 Machine Learning Pipeline:
    1. **Data Loading & Exploration**
    2. **Text Preprocessing & Cleaning**  
    3. **Feature Extraction (TF-IDF)**
    4. **Model Training with Class Balancing**
    5. **Performance Evaluation**
    6. **Real-time Prediction**
    
    ### 🚀 Technologies Used:
    - **Streamlit**: Web application framework
    - **Scikit-learn**: Machine learning models and preprocessing
    - **NLTK**: Natural language processing
    - **Pandas**: Data manipulation
    - **Matplotlib/Seaborn**: Data visualization
    - **WordCloud**: Text visualization
    
    ### 📈 Model Performance:
    The application handles imbalanced datasets effectively using:
    - **Balanced class weights** in Logistic Regression
    - **Stratified train-test split** to maintain class distribution
    - **Multiple evaluation metrics** (Accuracy, Precision, Recall, F1-Score)
    - **Confusion matrices** for detailed performance analysis
    
    ### 🔒 Best Practices Implemented:
    - **Robust text preprocessing** pipeline
    - **Cross-validation** ready architecture
    - **Model comparison** and selection
    - **Professional UI/UX** design
    - **Error handling** and validation
    - **Modular code** structure
    
    ---
    
    **Created for educational and demonstration purposes** 🎓
    
    *For production use, consider additional features like model persistence, API endpoints, and advanced security measures.*
    """)

if __name__ == "__main__":
    main()