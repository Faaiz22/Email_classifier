import streamlit as st
import joblib
import os # Import os module to check for file existence

# Set the page configuration for a wider layout and a title
st.set_page_config(layout="centered", page_title="Email Spam Classifier")

# --- UI Elements (CSS for styling) ---
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(to bottom right, #e0f2fe, #ede9fe);
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stTextArea textarea {
        border-radius: 0.5rem;
        border: 1px solid #d1d5db;
        padding: 1rem;
        font-size: 1.125rem; /* text-lg */
        color: #374151; /* gray-700 */
        min-height: 150px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    .stButton > button {
        width: 100%;
        padding: 0.75rem 1.5rem; /* py-3 px-6 */
        border-radius: 0.5rem; /* rounded-lg */
        background-color: #2563eb; /* blue-600 */
        color: white;
        font-weight: bold;
        font-size: 1.125rem; /* text-lg */
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: none;
    }
    .stButton > button:hover:enabled {
        background-color: #1d4ed8; /* blue-700 */
        transform: scale(1.02);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    .stButton > button:active:enabled {
        transform: scale(0.98);
    }
    .stButton > button:disabled {
        background-color: #9ca3af; /* gray-400 */
        cursor: not-allowed;
        box-shadow: none;
    }
    .result-box {
        padding: 1.25rem; /* p-5 */
        border-radius: 0.5rem; /* rounded-lg */
        text-align: center;
        font-weight: 600; /* font-semibold */
        font-size: 1.25rem; /* text-xl */
        margin-top: 2rem; /* mt-8 */
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.06); /* shadow-inner */
    }
    .spam-result {
        background-color: #fee2e2; /* red-100 */
        color: #b91c1c; /* red-700 */
        border: 1px solid #fca5a5; /* red-300 */
    }
    .ham-result {
        background-color: #dcfce7; /* green-100 */
        color: #16a34a; /* green-700 */
        border: 1px solid #86efac; /* green-300 */
    }
    .error-box {
        padding: 1rem; /* p-4 */
        background-color: #fee2e2; /* red-100 */
        border: 1px solid #ef4444; /* red-400 */
        color: #b91c1c; /* red-700 */
        border-radius: 0.5rem; /* rounded-lg */
        text-align: center;
        font-weight: 500; /* font-medium */
        margin-top: 2rem; /* mt-8 */
    }
    .footer-text {
        font-size: 0.75rem; /* text-xs */
        color: #6b7280; /* gray-500 */
        text-align: center;
        margin-top: 2rem; /* mt-8 */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h1 style='text-align: center; font-size: 2.5rem; font-weight: 800; color: #374151;'>
        <span style='color: #2563eb;'>Email</span> <span style='color: #8b5cf6;'>Spam</span> Classifier
    </h1>
    <p style='text-align: center; color: #4b5563; margin-bottom: 2rem;'>
        Paste your email content below to check if it's spam or ham.
    </p>
    """,
    unsafe_allow_html=True
)

# --- Model Loading ---
# Define paths to the pre-trained model and vectorizer
# Assuming they are in an 'app/' subdirectory as per your GitHub structure
MODEL_PATH = 'app/spam_classifier.pkl'
VECTORIZER_PATH = 'app/vectorizer.pkl'

# Use Streamlit's caching to load the model and vectorizer only once
@st.cache_resource
def load_assets():
    try:
        # Check if files exist before loading
        if not os.path.exists(MODEL_PATH):
            st.error(f"Error: Model file not found at {MODEL_PATH}. Please ensure it's in the correct directory.")
            return None, None
        if not os.path.exists(VECTORIZER_PATH):
            st.error(f"Error: Vectorizer file not found at {VECTORIZER_PATH}. Please ensure it's in the correct directory.")
            return None, None

        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        st.success("Model and vectorizer loaded successfully!")
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {e}")
        return None, None

classifier_model, text_vectorizer = load_assets()

# Check if assets were loaded successfully before proceeding
if classifier_model is None or text_vectorizer is None:
    st.stop() # Stop the app if assets couldn't be loaded

# --- User Input ---
email_content = st.text_area(
    "Enter your email content here...",
    height=200,
    placeholder="e.g., 'URGENT! You've won $1,000,000! Click here NOW to claim your prize!'"
)

# --- Classification Logic ---
if st.button("Classify Email", disabled=not email_content.strip()):
    with st.spinner("Classifying..."):
        try:
            # 1. Preprocess the input text using the loaded vectorizer
            # The vectorizer expects an iterable (e.g., a list) of strings
            vectorized_text = text_vectorizer.transform([email_content])

            # 2. Make a prediction using the loaded classifier model
            prediction = classifier_model.predict(vectorized_text)[0]

            # 3. Map the numerical prediction to 'ham' or 'spam'
            # Assuming 0 for ham and 1 for spam, adjust if your model uses different labels
            if prediction == 1: # Assuming 1 is for spam
                st.markdown(f"<div class='result-box spam-result'><p><span style='font-weight: bold;'>Result:</span> Spam</p></div>", unsafe_allow_html=True)
            else: # Assuming 0 is for ham
                st.markdown(f"<div class='result-box ham-result'><p><span style='font-weight: bold;'>Result:</span> Ham</p></div>", unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f"<div class='error-box'><p>An error occurred during classification: {e}</p></div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown(
    """
    <p class='footer-text'>
        Using a pre-trained machine learning model.
    </p>
    """,
    unsafe_allow_html=True
)
