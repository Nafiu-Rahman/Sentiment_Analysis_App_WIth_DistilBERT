import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "N4F1U/sentiment-analysis-distilbert"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

labels = ["Negative", "Positive"]

def predict_sentiment(review):
    inputs = tokenizer(review, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return {labels[0]: float(probs[0][0]), labels[1]: float(probs[0][1])}

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=4, placeholder="Enter a movie review here..."),
    outputs=gr.Label(num_top_classes=2),
    title="🎬 Sentiment Analysis with DistilBERT",
    description="Type a movie review to classify it as Positive or Negative."
)

if __name__ == "__main__":
    demo.launch()