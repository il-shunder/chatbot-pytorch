import json
import os
import random
from datetime import datetime

import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from torch.utils.data import DataLoader, TensorDataset

nltk.download("punkt_tab", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("stopwords", quiet=True)


MODEL_PATH = "assistant.pth"
DIMENSIONS_PATH = "dimensions.json"


class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ChatbotAssistant:
    def __init__(self, intents_path: str, method_mappings: dict = {}, remove_stopwords=False, use_synonyms=False):
        self.model = None
        self.intents_path = intents_path
        self.method_mappings = method_mappings

        self.stop_words = stopwords.words("english") if remove_stopwords else None
        self.use_synonyms = use_synonyms

        self.lemmatizer = WordNetLemmatizer()

        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}

        self.X = None
        self.y = None

    def tokenize_and_lemmatize(self, text):
        words = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word.lower()) for word in words]

        if self.stop_words:
            tokens = [token for token in tokens if token not in self.stop_words]

        if self.use_synonyms:
            tagged = pos_tag(tokens)

            expanded = set()
            for word, tag in tagged:
                wn_pos = self.get_wordnet_pos(tag)
                lemma = (
                    self.lemmatizer.lemmatize(word.lower(), pos=wn_pos)
                    if wn_pos
                    else self.lemmatizer.lemmatize(word.lower())
                )
                expanded.add(lemma)

                if wn_pos:
                    expanded.update(self.get_top_synonyms(lemma, pos=wn_pos))

            return list(expanded)
        return tokens

    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        if os.path.exists(self.intents_path):
            intents_data = json.loads(open(self.intents_path).read())

            for intent in intents_data["intents"]:
                if intent["tag"] not in self.intents:
                    self.intents.append(intent["tag"])
                    self.intents_responses[intent["tag"]] = intent["responses"]

                for pattern in intent["patterns"]:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, intent["tag"]))

            self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        bags = []
        indices = []

        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words)
            intent_index = self.intents.index(document[1])

            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(indices)

    def train_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatbotModel(self.X.shape[1], len(self.intents))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=self.model.parameters(), lr=lr)

        self.model.train()

        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X, batch_y in loader:
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                running_loss += loss

            print(f"Epoch: {epoch + 1}, Loss: {running_loss / len(loader):.4f}")

    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)

        with open(dimensions_path, "w") as f:
            json.dump({"input_size": self.X.shape[1], "output_size": len(self.intents)}, f)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, "r") as f:
            dimensions = json.load(f)

        self.model = ChatbotModel(dimensions["input_size"], dimensions["output_size"])
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

    def process_message(self, message):
        words = self.tokenize_and_lemmatize(message)
        bag = self.bag_of_words(words)

        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)

            predicted_class_index = torch.argmax(predictions, dim=1).item()
            prediction_intent = self.intents[predicted_class_index]

            if self.method_mappings:
                if prediction_intent in self.method_mappings:
                    self.method_mappings[prediction_intent]()

            if self.intents_responses[prediction_intent]:
                return random.choice(self.intents_responses[prediction_intent])
            else:
                return None

    def get_wordnet_pos(self, tag):
        return {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}.get(tag[0], None)

    def get_top_synonyms(self, word, pos=None, topn=3):
        synsets = wordnet.synsets(word, pos=pos) if pos else wordnet.synsets(word)
        synonyms = {
            lemma.name().lower().replace("_", " ")
            for syn in synsets
            for lemma in syn.lemmas()
            if lemma.name().lower().replace("_", " ") != word
        }

        scored = [(syn, wordnet.synsets(syn)[0].lemmas()[0].count() if wordnet.synsets(syn) else 0) for syn in synonyms]
        return [syn for syn, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:topn]]


def get_time():
    current_time = datetime.now().strftime("%I:%M %p")
    print(f"It is {current_time}")


def get_date():
    current_date = datetime.now().strftime("%Y-%m-%d")
    print(f"Today's date is {current_date}")


def get_stock():
    stocks = ["AAPL", "GOOGL", "MSFT"]
    print(random.choice(stocks))


if __name__ == "__main__":
    assistant = None

    if not os.path.exists(MODEL_PATH):
        assistant = ChatbotAssistant(
            "intents.json", method_mappings={"get_time": get_time, "get_date": get_date, "get_stock": get_stock}
        )
        assistant.parse_intents()
        assistant.prepare_data()
        assistant.train_model(batch_size=8, lr=0.001, epochs=100)

        assistant.save_model(MODEL_PATH, DIMENSIONS_PATH)
    else:
        assistant = ChatbotAssistant(
            "intents.json", method_mappings={"get_time": get_time, "get_date": get_date, "get_stock": get_stock}
        )
        assistant.parse_intents()
        assistant.load_model(MODEL_PATH, DIMENSIONS_PATH)

    if assistant:
        while True:
            message = input("Enter your message (to quit, enter '/quit'): ")

            if message == "/quit":
                break

            print(assistant.process_message(message))
