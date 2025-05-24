import streamlit as st
import requests
import json

# Set the page configuration for a wider layout and a title
st.set_page_config(layout="centered", page_title="Email Spam Classifier")

# --- UI Elements ---
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
    .unknown-result {
        background-color: #fffbeb; /* yellow-100 */
        color: #b45309; /* yellow-700 */
        border: 1px solid #fcd34d; /* yellow-300 */
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

# Text area for user input
email_content = st.text_area(
    "Enter your email content here...",
    height=200,
    placeholder="e.g., 'URGENT! You've won $1,000,000! Click here NOW to claim your prize!'"
)

# Button to trigger classification
if st.button("Classify Email", disabled=not email_content.strip()):
    # Display a spinner while loading
    with st.spinner("Classifying..."):
        # Construct the prompt for the LLM
        prompt = f"Classify the following email content as either 'ham' or 'spam'. Respond with only 'ham' or 'spam'.\n\nEmail: \"{email_content}\""

        try:
            # Prepare the payload for the Gemini API request
            chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
            payload = {"contents": chat_history}

            # Define the API key (empty string for Canvas environment) and API URL
            # In a real Streamlit app, you'd load this from st.secrets or environment variables
            api_key = "" # Canvas will provide this at runtime
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

            # Make the API call to Gemini
            response = requests.post(api_url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))

            # Check if the API response was successful
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

            result = response.json()

            # Process the LLM's response
            if result.get("candidates") and len(result["candidates"]) > 0 and \
               result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts") and \
               len(result["candidates"][0]["content"]["parts"]) > 0:
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                # Normalize the response to 'ham' or 'spam'
                if "spam" in text.lower():
                    st.markdown(f"<div class='result-box spam-result'><p><span style='font-weight: bold;'>Result:</span> Spam</p></div>", unsafe_allow_html=True)
                elif "ham" in text.lower():
                    st.markdown(f"<div class='result-box ham-result'><p><span style='font-weight: bold;'>Result:</span> Ham</p></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='result-box unknown-result'><p>Unable to classify. Please try again or refine your input.</p></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='error-box'><p>No valid classification found in the response.</p></div>", unsafe_allow_html=True)

        except requests.exceptions.HTTPError as http_err:
            st.markdown(f"<div class='error-box'><p>API Error: {http_err}</p><p>Details: {response.text}</p></div>", unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f"<div class='error-box'><p>An unexpected error occurred: {e}</p></div>", unsafe_allow_html=True)

# Footer text
st.markdown(
    """
    <p class='footer-text'>
        Powered by Gemini 2.0 Flash
    </p>
    """,
    unsafe_allow_html=True
)
