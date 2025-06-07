import numpy as np
import pickle
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import pandas as pd

app = Flask(__name__)

#model = pickle.load(open("./ufo-model.pkl", "rb"))
# Load dataset (Ensure this is correctly linked)
ufos = pd.read_csv('./data/ufos.csv') 

ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
ufos.dropna(inplace=True)
ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]

# Encode countries
label_encoder = LabelEncoder()
ufos["Country"] = label_encoder.fit_transform(ufos["Country"])
country_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# Select features globally
selected_features = ["Seconds", "Latitude", "Longitude"]
X = ufos[selected_features]
y = ufos["Country"]


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/train", methods=["GET"])
def train():
    """Train model when this route is accessed."""
    global model  # Make model accessible throughout the app
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Save trained model
    with open("./ufo-model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
    
    return f"Model trained successfully! Accuracy: {accuracy:.2f}"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Convert input values to float instead of int
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)

        output = prediction[0]
        countries = ["Australia", "Canada", "Germany", "UK", "US"]

        # Ensure the output index is within the list bounds
        if 0 <= output < len(countries):
            return render_template("index.html", prediction_text="Likely country: {}".format(countries[output]))
        else:
            return render_template("index.html", prediction_text="Error: Prediction index out of bounds.")

    except ValueError:
        return render_template("index.html", prediction_text="Error: Invalid input format. Please enter valid numbers.")


if __name__ == "__main__":
    app.run(debug=True)