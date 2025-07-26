from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from fastapi.middleware.cors import CORSMiddleware
import os # Import os to handle file paths robustly

# Define email input schema
class EmailInput(BaseModel):
    email: str

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load the model and vectorizer
# Use os.path.join for robust path handling, crucial for FileNotFoundError if CWD is tricky
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')

with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

with open(VECTORIZER_PATH, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# --- START: NEW THRESHOLD LOGIC ---
# Define the custom threshold for classifying as phishing.
# This value should be between 0 and 1.
# - A higher value (e.g., 0.7 or 0.8) makes the model *less* likely to classify as phishing (fewer false positives).
# - A lower value (e.g., 0.3 or 0.4) makes the model *more* likely to classify as phishing.
# You will need to experiment with this value! Start with 0.6 or 0.7.
CUSTOM_PHISHING_THRESHOLD = 0.7 # <<<--- EXPERIMENT WITH THIS VALUE (e.g., 0.6, 0.7, 0.8)

# Define the predict endpoint
@app.post("/predict")
async def predict(email: EmailInput):
    # Vectorize the input email text
    email_text = email.email
    email_vector = vectorizer.transform([email_text])

    # Get prediction probabilities
    # prediction_proba will be an array like [prob_class_0, prob_class_1]
    # where class 0 is "Safe" and class 1 is "Phishing"
    prediction_proba = model.predict_proba(email_vector)[0]

    phishing_probability = prediction_proba[1] # Probability of being 'Phishing Email'
    safe_probability = prediction_proba[0]    # Probability of being 'Safe Email'

    final_prediction_label: str
    final_confidence_value: float

    if phishing_probability >= CUSTOM_PHISHING_THRESHOLD:
        final_prediction_label = "Phishing Email"
        final_confidence_value = phishing_probability
    else:
        # If it's below the phishing threshold, it's considered safe.
        final_prediction_label = "Safe Email"
        final_confidence_value = safe_probability # Use the safe probability as confidence

    # Return the prediction result and confidence
    return {
        "prediction": final_prediction_label,
        "confidence": float(final_confidence_value)
    }
# --- END: NEW THRESHOLD LOGIC ---