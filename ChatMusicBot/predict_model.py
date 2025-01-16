import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Hàm chuẩn hóa dữ liệu
def standardize_data(row):
    row = re.sub(r"[.,?]+$", "", row)
    row = row.replace(",", " ").replace(".", " ").replace(";", " ").replace("“", " ").replace(":", " ").replace("”", " ").replace('"', " ").replace("'", " ").replace("!", " ").replace("?", " ").replace("-", " ").replace("?", " ")
    row = row.strip().lower()
    return row

# Hàm tải mô hình mới
def load_bert():
    model = AutoModelForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")
    tokenizer = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", use_fast=False)
    return model, tokenizer

# Hàm dự đoán cảm xúc
def predict_sentiment(text):
    """Dự đoán cảm xúc của một đoạn văn bản."""
    # Tải PhoBert và tokenizer
    model, tokenizer = load_bert()
    model.eval()  # Đặt chế độ đánh giá

    # Chuẩn hóa văn bản đầu vào
    processed_text = standardize_data(text)

    # Tokenizer văn bản
    inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, padding=True, max_length=256)

    # Dự đoán cảm xúc
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Tìm nhãn dự đoán có xác suất cao nhất
    prediction = torch.argmax(probabilities, dim=-1).item()
    confidence = torch.max(probabilities).item()

    # Ánh xạ nhãn số nguyên thành chuỗi cảm xúc
    label_mapping = {0: "buồn", 1: "vui", 2: "bình thường"}
    sentiment = label_mapping[prediction]

    print(f"Sentiment: {sentiment}, Confidence: {confidence}")
    return sentiment
if __name__ == "__main__":
    # Example usage
    text = "Hãy cho tôi một bài hát tâm trạng buồn"
    sentiment, confidence = predict_sentiment(text)
    print(f"Sentiment: {sentiment}, Confidence: {confidence}")
