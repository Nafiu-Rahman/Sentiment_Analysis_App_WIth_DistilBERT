# 🎬 Sentiment Analysis with DistilBERT

This project is a Sentiment Analysis App that classifies movie reviews as **Positive** or **Negative** using a fine-tuned **DistilBERT** model and a Gradio frontend.

---

##  Project Overview
The goal of this project is to create an interactive web application that allows users to input movie reviews and receive sentiment predictions in real-time. The model is fine-tuned on a publicly available movie review dataset (IMDb).

- **Model:** DistilBERT (pretrained and fine-tuned)
- **Frontend:** Gradio interface
- **Backend:** Hugging Face Transformers, PyTorch

---

##  Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Nafiu-Rahman/Sentiment_Analysis_App_With_DistilBERT
cd Sentiment_Analysis_App_With_DistilBERT
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app locally
```bash
python app.py
```

### Dependencies

- transformers==4.34.0
- torch==2.2.0
- gradio==3.49.0
- numpy

### Live Demo

The app is deployed on Hugging Face Spaces:

[Sentiment Analysis App](https://huggingface.co/spaces/N4F1U/sentiment-app)

### Dataset
The model was fine-tuned on the IMDb movie review dataset.
https://ai.stanford.edu/~amaas/data/sentiment/

### Demo video
https://youtu.be/zldKd4xRrGI

### License
This project is open source and free to use for educational purposes.


