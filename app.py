import streamlit as st
import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.applications.resnet import preprocess_input
from PIL import Image

# 🔹 Load trained model
model = tf.keras.models.load_model("best_model.h5")
classes = ["Rock", "Paper", "Scissors"]


# 🔹 Prediction function
def predict(image):
    # Convert PIL image to array
    image = np.array(image)

    # Resize to training size
    image_resized = tf.image.resize(image, (300, 300))
    image_resized = preprocess_input(image_resized)

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
st.write("Upload an image OR take a live photo of your hand showing Rock ✊, Paper ✋, or Scissors ✌️.")


# Option 1: Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Option 2: Take a live photo
camera_file = st.camera_input("Or take a live photo")

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
elif camera_file is not None:
    image = Image.open(camera_file).convert("RGB")

if image is not None:
    st.image(image, caption="Input Image", use_container_width=True)

    # Predict
    user_choice, computer_choice, result = predict(image)

    # Show results
    st.write(f"**Your Choice:** {user_choice}")
    st.write(f"**Computer's Choice:** {computer_choice}")
    st.write(f"**Result:** {result}")
