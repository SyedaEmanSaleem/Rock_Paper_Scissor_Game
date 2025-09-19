import streamlit as st
import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.applications.resnet import preprocess_input
from PIL import Image

# 🔹 Load trained model
model = tf.keras.models.load_model("best_model.h5")

# ✅ Must match TFDS label order: 0=Rock, 1=Paper, 2=Scissors
classes = ["Rock", "Paper", "Scissors"]


def predict(image):
    # Convert PIL image to array
    image = np.array(image)

    # Resize to training size
    image_resized = tf.image.resize(image, (300, 300))

    # Rescale like in training
    image_resized = image_resized / 255.0  

    # Add batch dimension
    image_resized = np.expand_dims(image_resized, axis=0)

    # Predict
    pred = model.predict(image_resized)
    user_choice = classes[np.argmax(pred)]

    # Computer random choice
    computer_choice = random.choice(classes)

    # Decide result
    if user_choice == computer_choice:
        result = "Draw!"
    elif (user_choice == "Rock" and computer_choice == "Scissors") or \
            (user_choice == "Paper" and computer_choice == "Rock") or \
            (user_choice == "Scissors" and computer_choice == "Paper"):
        result = "You Win!"
    else:
        result = "You Lose!"

    return user_choice, computer_choice, result



# 🔹 Streamlit interface
st.title("Rock Paper Scissors Game 🎮")
st.write("Upload an image of your hand showing Rock ✊, Paper ✋, or Scissors ✌️.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Predict
    user_choice, computer_choice, result, probs = predict(image)

    # Show results
    st.write(f"**Your Choice:** {user_choice}")
    st.write(f"**Computer's Choice:** {computer_choice}")
    st.write(f"**Result:** {result}")

    # Debugging: show prediction probabilities
    st.write("### Prediction Probabilities")
    st.json(probs)
