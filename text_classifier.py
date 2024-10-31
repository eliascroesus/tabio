import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import urllib.parse
from tqdm import tqdm
import csv
from googletrans import Translator
import langdetect

class WebTextClassifier:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = None
        self.categories = []
        self.translator = Translator()

    def prepare_data(self, texts, labels):
        return WebTextDataset(texts, labels, self.tokenizer)

    def train_model(self, train_dataloader, val_dataloader, epochs=20):
        num_labels = len(self.categories)
        if not self.model:
            self.model = DistilBertForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels)
        else:
            print("Continuing training with existing model.")
            # Ensure the existing model has the correct number of labels
            if self.model.num_labels != num_labels:
                print(f"Reinitializing classification layer for {num_labels} labels")
                self.model.num_labels = num_labels
                self.model.classifier = torch.nn.Linear(self.model.config.dim, num_labels)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)  # Smaller learning rate
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            
            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Average training loss: {avg_train_loss:.4f}")
            
            self.model.eval()
            val_accuracy = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    val_accuracy += (predictions == labels).float().mean().item()
            
            val_accuracy /= len(val_dataloader)
            print(f"Validation Accuracy: {val_accuracy:.4f}")

    def save_model(self, filename):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
            'categories': self.categories
        }, filename)

    def load_model(self, filename):
        checkpoint = torch.load(filename)
        self.categories = checkpoint['categories']
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=len(self.categories),
            state_dict=checkpoint['model_state_dict']
        )
        self.tokenizer = checkpoint['tokenizer']

    def preprocess_url(self, url):
        parsed_url = urllib.parse.urlparse(url)
        domain = parsed_url.netloc
        path = parsed_url.path
        query = parsed_url.query
        return f"{domain} {path.replace('/', ' ')} {query.replace('&', ' ')}"

    def translate_if_needed(self, text):
        try:
            lang = langdetect.detect(text)
            if lang != 'en':
                # Instead of translating, we'll just return the original text
                # You might want to add some handling for non-English text here
                return text
            return text
        except:
            return text

    def classify(self, text, allowed_categories):
        self.model.eval()
        text = self.translate_if_needed(text)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits.squeeze()
        
        # Get probabilities for all categories
        probabilities = torch.softmax(logits, dim=0)
        
        # Filter to only include allowed categories
        allowed_indices = [self.categories.index(cat) for cat in allowed_categories if cat in self.categories]
        filtered_probs = probabilities[allowed_indices]
        
        # Get the top 3 predictions
        top_3_probs, top_3_indices = torch.topk(filtered_probs, min(3, len(filtered_probs)))
        
        # Convert to actual category names
        top_3_categories = [allowed_categories[i] for i in top_3_indices]
        
        # Use some heuristics for better classification
        if 'Business' in top_3_categories and any(keyword in text.lower() for keyword in ['marketing', 'business', 'money', 'finance', 'crypto']):
            return 'Business'
        elif 'Entertainment' in top_3_categories and any(keyword in text.lower() for keyword in ['music', 'video', 'movie', 'song', 'artist']):
            return 'Entertainment'
        elif 'Technology' in top_3_categories and any(keyword in text.lower() for keyword in ['tech', 'software', 'hardware', 'programming']):
            return 'Technology'
        
        # If no heuristics match, return the top prediction
        return top_3_categories[0]

class WebTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def map_to_main_category(category):
    main_categories = {
        'Business': ['Business', 'Finance', 'Cryptocurrency', 'Stock_trading'],
        'Technology': ['Technology', 'Computer_science', 'Artificial_intelligence', 'gadgets'],
        'Education': ['Education', 'Online_tutorials', 'E-learning', 'Science'],
        'Entertainment': ['Entertainment', 'Film', 'Music', 'gaming', 'Arts'],
        'News': ['News', 'Politics', 'worldnews', 'Current_events'],
        'Sports': ['Sports', 'Olympic_sports', 'Extreme_sports'],
        'Food': ['Food', 'Cooking', 'recipes'],
        'Travel': ['Travel', 'backpacking'],
        'Health': ['Health', 'Fitness', 'Nutrition', 'Mental_health'],
        'Other': ['Other', 'DIY', 'lifehacks', 'philosophy', 'Futurology']
    }
    
    for main_cat, sub_cats in main_categories.items():
        if category in sub_cats:
            return main_cat
    return 'Other'  # Default to 'Other' if no match is found

def load_and_preprocess_data(filename):
    df = pd.read_csv(filename)
    df['Category'] = df['Category'].apply(map_to_main_category)
    texts = df['URL'] + ' ' + df['Title']
    categories = df['Category'].unique()
    category_to_index = {cat: i for i, cat in enumerate(categories)}
    labels = df['Category'].map(category_to_index)
    
    print(f"Number of unique categories: {len(categories)}")
    print(f"Categories: {categories}")
    print(f"Max label value: {labels.max()}")
    
    return texts.tolist(), labels.tolist(), categories.tolist()

def main():
    print("Welcome to the Web Text Classifier!")
    
    classifier = WebTextClassifier()
    
    # Main categories
    main_categories = ['Business', 'Technology', 'Education', 'Entertainment', 'News', 'Sports', 'Food', 'Travel', 'Health', 'Other']
    
    # Always train the model
    print("Loading and preprocessing data...")
    texts, labels, categories = load_and_preprocess_data('web_classification_dataset.csv')
    
    classifier.categories = main_categories

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    train_dataset = classifier.prepare_data(train_texts, train_labels)
    val_dataset = classifier.prepare_data(val_texts, val_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32)

    print("Training the model...")
    classifier.train_model(train_dataloader, val_dataloader, epochs=20)  # Increased epochs for better training

    print("Saving the model...")
    classifier.save_model('web_classifier.pth')
    print("Model trained and saved.")

    # Test the model
    while True:
        url = input("Enter a URL to classify (or 'quit' to exit): ")
        if url.lower() == 'quit':
            break
        tab_name = input("Enter the tab name: ")
        allowed_categories = main_categories  # Use all categories for testing
        text_to_classify = classifier.preprocess_url(url) + " " + tab_name
        predicted_category = classifier.classify(text_to_classify, allowed_categories)
        print(f"Predicted category: {predicted_category}")

if __name__ == "__main__":
    main()
