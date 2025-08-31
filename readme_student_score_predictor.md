# 🎓 Student Exam Score Predictor

A futuristic Streamlit web app that predicts student exam scores based on uploaded datasets or manual input. The app automatically cleans your data, detects the exam score column, and trains a linear regression model to make predictions. Perfect for teachers, students, or anyone looking to estimate performance! 🚀

---

## ⚡ Features

- **CSV Upload**: Upload any dataset with student information.
- **Auto Data Cleaning**: Removes useless columns, fills missing values, and drops empty columns.
- **Automatic Target Detection**: Detects columns like `exam`, `score`, `marks`, `result`, or `grade` as the target.
- **Linear Regression Prediction**: Trains a model and predicts exam scores based on input features.
- **User-Friendly Input Form**: Input student information via sliders and dropdowns.
- **Futuristic UI**: Gradient backgrounds, glowing headers, and animated buttons for a modern look.

---

## 🛠 Installation

1. Clone the repository:
```bash
git clone https://github.com/abdullahrauf45/Student-score-predictor.git
cd student-score-predictor
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

> **requirements.txt**
```
streamlit
pandas
scikit-learn
```

---

## 🚀 Usage

Run the app with:

```bash
streamlit run app.py
```

1. Upload a CSV dataset with student features and exam scores.
2. Review the auto-cleaned dataset preview.
3. The app will automatically detect the exam score column (or let you select it).
4. Enter student information in the input form.
5. Click **Predict Exam Score** to see the predicted result.

---

## 📁 Dataset Guidelines

- CSV format.
- Include features like study hours, attendance, or other student metrics.
- Target column can be named `exam`, `score`, `marks`, `result`, or `grade`.
- Avoid completely empty columns or irrelevant IDs/names (though the app auto-removes them).

---

## 🎨 Styling & UI

- Futuristic gradient background with animated color shift.
- Glowing headers and buttons.
- Modern font: Orbitron.
- Smooth hover effects on buttons.

---

## 🤖 Model Details

- **Algorithm**: Linear Regression
- **Preprocessing**:
  - Numeric columns: missing values filled with mean.
  - Categorical columns: Label Encoding.
  - Irrelevant columns dropped automatically.
- **Train/Test Split**: 80/20

---

## 📌 Future Improvements

- Add other regression models like Random Forest, XGBoost for better accuracy.
- Allow saving/loading trained models for faster predictions.
- Interactive visualization of predictions vs actual scores.

---

## 📜 License

MIT License © 2025 Abdullah Rauf

---

## 🎯 Screenshots

![Upload CSV](screenshots/upload_csv.png)  
![Prediction](screenshots/prediction.png)

---

> Created with ❤️ using Streamlit and Python

