import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QLineEdit, QComboBox,
    QPushButton, QVBoxLayout, QWidget, QMessageBox
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# Charger les données pour entraîner le modèle
df = pd.read_csv('car_data.csv')

# Prétraitement des données avec des encodeurs séparés
fuel_type_encoder = LabelEncoder()
selling_type_encoder = LabelEncoder()
transmission_encoder = LabelEncoder()

df['Fuel_Type'] = fuel_type_encoder.fit_transform(df['Fuel_Type'])
df['Selling_type'] = selling_type_encoder.fit_transform(df['Selling_type'])
df['Transmission'] = transmission_encoder.fit_transform(df['Transmission'])
df = df.drop('Car_Name', axis=1)

X = df[['Year', 'Driven_kms', 'Fuel_Type', 'Selling_type', 'Transmission', 'Owner']]
y = df['Selling_Price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entraîner un modèle Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Sauvegarder le modèle et les encodeurs
joblib.dump(model, 'car_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(fuel_type_encoder, 'fuel_type_encoder.pkl')
joblib.dump(transmission_encoder, 'transmission_encoder.pkl')


# Classe principale avec PyQt5
class CarPricePredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Charger les modèles et encodeurs
        self.model = joblib.load('car_price_model.pkl')
        self.scaler = joblib.load('scaler.pkl')
        self.fuel_type_encoder = joblib.load('fuel_type_encoder.pkl')
        self.transmission_encoder = joblib.load('transmission_encoder.pkl')

        # Configuration de la fenêtre principale
        self.setWindowTitle("Car Price Prediction")
        self.setGeometry(300, 200, 500, 400)
        self.setStyleSheet("background-color: #f2f2f2;")

        # Widgets et Layout
        layout = QVBoxLayout()

        self.title_label = QLabel("Car Price Prediction")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("Arial", 18, QFont.Bold))
        self.title_label.setStyleSheet("color: #2c3e50; margin-bottom: 20px;")
        layout.addWidget(self.title_label)

        self.label_year = QLabel("Year of car design:")
        self.label_year.setStyleSheet("color: #34495e;")
        layout.addWidget(self.label_year)
        self.input_year = QLineEdit()
        self.input_year.setStyleSheet("background-color: #ecf0f1; border: 1px solid #bdc3c7; padding: 5px;")
        layout.addWidget(self.input_year)

        self.label_kms = QLabel("Mileage traveled (km):")
        self.label_kms.setStyleSheet("color: #34495e;")
        layout.addWidget(self.label_kms)
        self.input_kms = QLineEdit()
        self.input_kms.setStyleSheet("background-color: #ecf0f1; border: 1px solid #bdc3c7; padding: 5px;")
        layout.addWidget(self.input_kms)

        self.label_fuel = QLabel("Fuel type :")
        self.label_fuel.setStyleSheet("color: #34495e;")
        layout.addWidget(self.label_fuel)
        self.combo_fuel = QComboBox()
        self.combo_fuel.addItems(["Petrol", "Diesel", "CNG"])
        self.combo_fuel.setStyleSheet("background-color: #ecf0f1; border: 1px solid #bdc3c7; padding: 5px;")
        layout.addWidget(self.combo_fuel)

        self.label_transmission = QLabel("Type de transmission :")
        self.label_transmission.setStyleSheet("color: #34495e;")
        layout.addWidget(self.label_transmission)
        self.combo_transmission = QComboBox()
        self.combo_transmission.addItems(["Manual", "Automatic"])
        self.combo_transmission.setStyleSheet("background-color: #ecf0f1; border: 1px solid #bdc3c7; padding: 5px;")
        layout.addWidget(self.combo_transmission)

        self.predict_button = QPushButton("Predict the price")
        self.predict_button.setStyleSheet(
            "background-color: #3498db; color: white; border-radius: 5px; padding: 10px; font-size: 14px;"
        )
        self.predict_button.clicked.connect(self.predict_price)
        layout.addWidget(self.predict_button)

        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.result_label.setStyleSheet("color: #27ae60; margin-top: 20px;")
        layout.addWidget(self.result_label)

        # Définir le layout principal
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def predict_price(self):
        try:
            # Récupérer les entrées utilisateur
            year = int(self.input_year.text())
            driven_kms = int(self.input_kms.text())
            fuel_type = self.combo_fuel.currentText()
            transmission = self.combo_transmission.currentText()

            # Encoder et normaliser les données
            fuel_type_encoded = self.fuel_type_encoder.transform([fuel_type])[0]
            transmission_encoded = self.transmission_encoder.transform([transmission])[0]

            features = np.array([[year, driven_kms, fuel_type_encoded, 0, transmission_encoded, 0]])
            features_scaled = self.scaler.transform(features)

            # Prédire le prix
            predicted_price = self.model.predict(features_scaled)[0]
            self.result_label.setText(f"Predicted price : ${predicted_price:.2f} M")

        except ValueError:
            QMessageBox.critical(self, "Error", "Please enter valid values ​​for year and mileage.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error has occurred: {str(e)}")


# Fonction principale
def main():
    app = QApplication(sys.argv)
    window = CarPricePredictionApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
