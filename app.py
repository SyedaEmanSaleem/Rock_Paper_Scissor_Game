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
    # Convert PIL image to NumPy array
    image = np.array(image)

    # Resize to training size
    image_resized = tf.image.resize(image, (300, 300))

    # 🔹 Match training: cast to float32 before preprocess_input
    image_resized = tf.cast(image_resized, tf.float32)

    # Apply same preprocessing as training
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
st.write("Upload an image of your hand showing Rock ✊, Paper ✋, or Scissors ✌️.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Predict
    user_choice, computer_choice, result = predict(image)

    # Show results
    st.write(f"**Your Choice:** {user_choice}")
    st.write(f"**Computer's Choice:** {computer_choice}")
    st.write(f"**Result:** {result}")
