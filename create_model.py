import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

class ModelWrapper:
    def __init__(self, df):
        self.df = df
        self.model = None
        self.is_neural_net = False
        self.label_encoder = LabelEncoder()

    def preprocess_data(self):
        # Replace 'target' with your target column
        X = self.df.drop('legal_action_status', axis=1)
        y = self.df['legal_action_status']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    def train_random_forest(self):
        self.model = RandomForestClassifier()
        self.model.fit(self.X_train, self.y_train)

    def train_xgboost(self):
        self.label_encoder.fit(self.y_train)  # Fit the encoder
        encoded_y_train = self.label_encoder.transform(self.y_train)
        self.model = XGBClassifier()
        self.model.fit(self.X_train, encoded_y_train)

    def train_neural_network(self):
        self.is_neural_net = True
        self.model = Sequential([
            # Add your layers here. Example:
            Dense(64, activation='relu', input_shape=(self.X_train.shape[1],)),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid'),
            Dense(2, activation='softmax')  # Output for two classes
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(self.X_train, to_categorical(self.y_train), epochs=10, batch_size=32)

    def evaluate_model(self):
        if self.is_neural_net:
            loss, accuracy = self.model.evaluate(self.X_test, to_categorical(self.y_test))
            print(f"Neural Network - Loss: {loss}, Accuracy: {accuracy}")
        else:
            predictions = self.model.predict(self.X_test)
            # Since label_encoder is already fitted, we can use it to transform y_test
            self.label_encoder.fit(self.y_train)
            encoded_y_test = self.label_encoder.transform(self.y_test)
            print("Accuracy:", accuracy_score(encoded_y_test, predictions))
            print(classification_report(encoded_y_test, predictions))


    def save_model(self, path='model.pkl'):
        if self.is_neural_net:
            self.model.save(path)
        else:
            joblib.dump(self.model, path)

    def load_model(self, path='model.pkl'):
        if self.is_neural_net:
            self.model = load_model(path)
        else:
            self.model = joblib.load(path)

# Usage
df = pd.read_csv('file1.csv')  # Load your preprocessed DataFrame
model_wrapper = ModelWrapper(df)
model_wrapper.preprocess_data()

# Train different models
model_wrapper.train_random_forest()
model_wrapper.evaluate_model()
model_wrapper.save_model('random_forest_model.pkl')

model_wrapper.train_xgboost()
model_wrapper.evaluate_model()
model_wrapper.save_model('xgboost_model.pkl')

model_wrapper.train_neural_network()
model_wrapper.evaluate_model()
model_wrapper.save_model('neural_network_model.h5')
