import sqlite3
from flask import Flask, render_template, request
import pickle
import re
import string
import matplotlib.pyplot as plt
import os
import pandas as pd
from flask import send_file

app = Flask(__name__)

# Create database and table
conn = sqlite3.connect("history.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    news TEXT,
    result TEXT,
    confidence REAL
)
""")

conn.commit()
conn.close()

# Load saved model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))


# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# History page
@app.route("/history")
def history():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM history ORDER BY id DESC")
    records = cursor.fetchall()

    conn.close()

    return render_template("history.html", records=records)

@app.route("/dashboard")
def dashboard():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()

    # Count fake news
    cursor.execute("SELECT COUNT(*) FROM history WHERE result='Fake News'")
    fake_count = cursor.fetchone()[0]

    # Count real news
    cursor.execute("SELECT COUNT(*) FROM history WHERE result='Real News'")
    real_count = cursor.fetchone()[0]

    conn.close()

    # Create chart
    labels = ["Fake News", "Real News"]
    values = [fake_count, real_count]

    plt.figure(figsize=(6, 6))
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title("Prediction Distribution")

    # Save chart
    if not os.path.exists("static"):
        os.makedirs("static")

    plt.savefig("static/chart.png")
    plt.close()

    return render_template(
        "dashboard.html",
        fake_count=fake_count,
        real_count=real_count
    )
@app.route("/export")
def export():
    conn = sqlite3.connect("history.db")

    df = pd.read_sql_query("SELECT * FROM history", conn)

    conn.close()

    file_path = "prediction_history.csv"
    df.to_csv(file_path, index=False)

    return send_file(file_path, as_attachment=True)
# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]

    # Clean input text
    cleaned_news = clean_text(news)

    # Convert into TF-IDF vector
    vectorized_news = vectorizer.transform([cleaned_news])

    # Predict result
    prediction = model.predict(vectorized_news)

    # Get confidence score
    probability = model.predict_proba(vectorized_news)
    confidence = round(max(probability[0]) * 100, 2)

    # Final result
    result = "Real News" if prediction[0] == 1 else "Fake News"

    # Save prediction into database
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO history (news, result, confidence) VALUES (?, ?, ?)",
        (news, result, confidence)
    )

    conn.commit()
    conn.close()

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)