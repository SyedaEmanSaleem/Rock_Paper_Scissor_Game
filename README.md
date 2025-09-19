
# Rock_Paper_Scissors_Game 🎮

This is a **Rock-Paper-Scissors game** built using **TensorFlow** and **Streamlit**.  
Upload an image of your hand showing **Rock ✊, Paper ✋, or Scissors ✌️** and play against the computer, which randomly selects its choice. The game uses a trained CNN model (`best_model.h5`) achieving **100% validation accuracy** on the dataset.

---

## Features

- Upload an image of your hand to play.
- Computer makes a random choice.
- Displays the winner: You Win, You Lose, or Draw.
- Uses a pre-trained **ResNet50-based CNN model** for hand gesture recognition.
- Built with **Streamlit** for an interactive web interface.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/SyedaEmanSaleem/Rock_Paper_Scissors_Game.git
cd Rock_Paper_Scissors_Game
````

2. Install required packages:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the app locally:

```bash
streamlit run app.py
```

* A browser window will open.
* Upload an image of your hand showing Rock, Paper, or Scissors.
* See the computer's choice and the result.

---

## Files in Repository

* `app.py` → Streamlit app file.
* `best_model.h5` → Pre-trained CNN model.
* `requirements.txt` → Required Python packages.

---

## Deployment

You can deploy this app online using **Streamlit Community Cloud**:

1. Push the repository to GitHub.
2. Go to [https://share.streamlit.io](https://share.streamlit.io).
3. Select your repository and branch (`main`), and set `app.py` as the entry point.
4. Click **Deploy**.

