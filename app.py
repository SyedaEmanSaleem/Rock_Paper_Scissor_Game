import streamlit as st
import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet import preprocess_input
from PIL import Image
import tensorflow_datasets as tfds

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 300

# Load dataset info to get correct class names
_, info = tfds.load("rock_paper_scissors", split=["train"], with_info=True)
CLASS_NAMES = info.features["label"].names  # ['rock', 'paper', 'scissors']

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_trained_model():
    model = load_model("best_model.h5")
    return model

model = load_trained_model()

# -----------------------------
# GAME LOGIC
# -----------------------------
def get_winner(user_choice, computer_choice):
    if user_choice == computer_choice:
        return "It's a Draw! 🤝"
    elif (user_choice == "rock" and computer_choice == "scissors") or \
         (user_choice == "paper" and computer_choice == "rock") or \
         (user_choice == "scissors" and computer_choice == "paper"):
        return "You Win! 🎉"
    else:
        return "You Lose! 😢"

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Rock Paper Scissors Game", page_icon="✊✋✌️")

st.title("✊✋✌️ Rock, Paper, Scissors Game")
st.write("Upload an image of your hand gesture. The model will predict your move, the computer will make a move, and then we’ll see who wins!")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your Move", use_container_width=True)  # fixed warning

    # Preprocess
    img_array = np.array(image)
    img_resized = tf.image.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_preprocessed = preprocess_input(img_resized)
    img_expanded = np.expand_dims(img_preprocessed, axis=0)

    # Prediction
    preds = model.predict(img_expanded)
    predicted_index = np.argmax(preds)
    user_choice = CLASS_NAMES[predicted_index]
    confidence = np.max(preds) * 100

    # Computer's random choice
    computer_choice = random.choice(CLASS_NAMES)

    # Winner
    result = get_winner(user_choice, computer_choice)

    # Show results
    st.subheader("Game Results")
    st.write(f"**Your Move:** {user_choice} ({confidence:.2f}% confidence)")
    st.write(f"**Computer’s Move:** {computer_choice}")
    st.markdown(f"### 🎮 {result}")

    # Confidence bar chart
    st.bar_chart(dict(zip(CLASS_NAMES, preds[0])))

    # Debugging info (optional, can remove later)
    st.write("🔍 Debug Info:")
    st.write("Class order:", CLASS_NAMES)
    st.write("Predicted index:", predicted_index)
    st.write("Raw prediction scores:", preds[0])
