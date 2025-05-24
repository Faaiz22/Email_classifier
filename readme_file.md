#  Spam Email Classifier

A comprehensive machine learning web application for classifying emails as spam or ham (legitimate) using advanced text processing and multiple classification algorithms.

##  Project Overview

This project implements a binary classification system that distinguishes between spam and ham emails using machine learning techniques including Logistic Regression and Naive Bayes. The application handles imbalanced datasets effectively and provides a user-friendly interface for real-time email classification.

##  Features

- **Multiple ML Models**: Logistic Regression & Naive Bayes with performance comparison
- **Advanced Text Preprocessing**: URL removal, email cleaning, stemming, stopword removal
- **Class Imbalance Handling**: Balanced class weights for better performance on imbalanced data
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score with visualizations
- **Interactive UI**: Modern Streamlit interface with real-time classification
- **Performance Analysis**: Confusion matrices and detailed classification reports
- **Real-time Classification**: Test your own text with instant predictions

##  Live Demo

- **Streamlit App**: [Your Streamlit Cloud URL here]
- **GitHub Repository**: [Your GitHub Repository URL here]

##  Dataset Requirements

The application expects a CSV file with the following format:

```csv
label,message
ham,"Hello, how are you today?"
spam,"URGENT! Claim your prize now!"
ham,"Meeting scheduled for tomorrow at 3 PM"
spam,"Free money! Click here now!"
```

**Required columns:**
- `label`: Contains 'spam' or 'ham' classifications
- `message`: Contains the email text content

##  Installation & Setup

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Local Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/spam-email-classifier.git
   cd spam-email-classifier
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

### Requirements.txt

Create a `requirements.txt` file with the following dependencies:

```txt
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
nltk==3.8.1
wordcloud==1.9.2
plotly==5.15.0
```

##  Deployment on Streamlit Cloud

### Step 1: Prepare Your Repository

1. Ensure your repository contains:
   - `streamlit_app.py` (main application file)
   - `requirements.txt` (dependencies)
   - `README.md` (this file)

2. Make sure all files are committed and pushed to GitHub

### Step 2: Deploy on Streamlit Cloud

1. **Visit** [share.streamlit.io](https://share.streamlit.io)

2. **Connect your GitHub account** if not already connected

3. **Click "New app"** and provide:
   - Repository: `your-username/spam-email-classifier`
   - Branch: `main` (or your default branch)
   - Main file path: `streamlit_app.py`

4. **Click "Deploy"** and wait for the application to build

5. **Your app will be available** at: `https://your-app-name.streamlit.app`

### Troubleshooting Deployment

If deployment fails:
- Check that `requirements.txt` includes all necessary packages
- Ensure Python version compatibility
- Verify all import statements work correctly
- Check Streamlit Cloud logs for specific error messages

##  How to Use

### 1. Upload & Train Models
- Navigate to the "Upload & Train" section
- Upload your CSV file with spam/ham labels
- Configure column mappings if needed
- Click "Start Training" to train both models
- View dataset analysis and model performance

### 2. Classify Text
- Go to "Classify Text" section
- Enter text manually or upload a text file
- Get predictions from both models
- View confidence scores and probability distributions

### 3. Analyze Performance
- Visit "Model Performance" section
- Compare metrics across models
- View confusion matrices
- Examine detailed classification reports

##  Technical Architecture

### Data Preprocessing Pipeline

1. **Text Cleaning**:
   - Lowercase conversion
   - URL and email removal
   - Phone number extraction
   - Special character removal

2. **NLP Processing**:
   - Stopword removal
   - Porter stemming
   - Tokenization

3. **Feature Engineering**:
   - TF-IDF vectorization
   - N-gram extraction (1-2 grams)
   - Feature selection (top 5000 features)

### Machine Learning Models

1. **Logistic Regression**:
   - Balanced class weights
   - L2 regularization
   - liblinear solver
   - Max iterations: 1000

2. **Naive Bayes**:
   - Multinomial distribution
   - Laplace smoothing (alpha=1.0)
   - Suitable for text classification

### Performance Metrics

- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed prediction breakdown

##  Model Performance

The application implements several techniques to handle imbalanced datasets:

- **Balanced Class Weights**: Automatically adjusts for class imbalance
- **Stratified Sampling**: Maintains class distribution in train/test splits
- **Multiple Metrics**: Focus on precision, recall, and F1-score over accuracy
- **Visual Analysis**: Confusion matrices and probability distributions

##  User Interface Features

- **Modern Design**: Clean, professional interface
- **Interactive Elements**: Real-time predictions and visualizations
- **Responsive Layout**: Works on desktop and mobile devices
- **Progress Indicators**: Clear feedback during model training
- **Error Handling**: Graceful error messages and validation

##  Code Structure

```
spam-email-classifier/
├── streamlit_app.py          # Main application file
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore               # Git ignore file (optional)
```

### Key Classes and Functions

- `SpamClassifier`: Main ML pipeline class
- `preprocess_text()`: Text cleaning and preprocessing
- `train_models()`: Model training with class balancing
- `calculate_metrics()`: Performance evaluation
- Page functions: `home_page()`, `upload_and_train_page()`, etc.

##  Advanced Features

### Model Comparison
- Side-by-side performance comparison
- Consensus voting system
- Confidence score analysis

### Text Analysis
- Word cloud generation (optional)
- Feature importance visualization
- N-gram analysis

### Real-time Classification
- Instant predictions
- Probability distributions
- Model confidence scores

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- **Scikit-learn** team for excellent ML library
- **Streamlit** team for the amazing web framework
- **NLTK** contributors for natural language processing tools
- The open-source community for inspiration and resources

##  Contact

- **GitHub**: [Your GitHub Profile]
- **LinkedIn**: [Your LinkedIn Profile]
- **Email**: [Your Email Address]

---

##  Quick Start Commands

```bash
# Clone and setup
git clone https://github.com/your-username
