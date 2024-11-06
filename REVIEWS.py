from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 1. Carica il tokenizer e il modello fine-tunato
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# 2. Funzione per predire il sentiment per un batch di recensioni
def predict_batch_sentiment(reviews):
    results = []
    for review in reviews:
        inputs = tokenizer(review, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment_score = torch.argmax(predictions).item()
        
        # Interpretazione del sentiment
        if sentiment_score >= 3:
            sentiment = "Positive"
        elif sentiment_score == 2:
            sentiment = "Neutral"
        else:
            sentiment = "Negative"
        
        results.append((review, sentiment))
    return results

# 3. Leggi le recensioni dal file di testo
reviews_file_path = "C:\\Users\\Fabio\\Desktop\\BERT with WEBDRIVER\\reviews.txt"

try:
    with open(reviews_file_path, "r", encoding="utf-8") as file:
        reviews = [line.strip().strip('"') for line in file if line.strip()]  # Rimuove le virgolette e righe vuote

    # 4. Analizza le recensioni lette
    more_results = predict_batch_sentiment(reviews)

    # 5. Mostra i risultati
    for review, sentiment in more_results:
        print(f"Review: {review}\nSentiment: {sentiment}\n")

except FileNotFoundError:
    print(f"File non trovato: {reviews_file_path}")
except Exception as e:
    print(f"Si è verificato un errore: {e}")
