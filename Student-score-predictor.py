import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# ==========================
# Futuristic CSS Styling
# ==========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(270deg, #000000, #001f3f, #003366, #006699, #000000);
    background-size: 1000% 1000%;
    animation: AnimationBG 30s ease infinite;
    color: white;
}
@keyframes AnimationBG {
    0%{background-position:0% 50%}
    50%{background-position:100% 50%}
    100%{background-position:0% 50%}
}
header {
    background: linear-gradient(90deg, #001f3f, #0066ff, #00ffff);
    color: white !important;
    height: 60px;
    text-align: center;
    padding: 15px;
    font-size: 22px;
    font-weight: bold;
    font-family: 'Orbitron', sans-serif;
    text-shadow: 0px 0px 15px #00ffff;
    border-bottom: 2px solid #00ffff;
}
footer {visibility: hidden;}
h1, h2, h3 {
    font-family: 'Orbitron', sans-serif;
    color: #00ffff;
    text-shadow: 0 0 20px #00ffff;
}
.stButton>button {
    background: linear-gradient(90deg, #00ffff, #0066ff);
    color: black;
    border-radius: 12px;
    font-weight: bold;
    box-shadow: 0px 0px 15px #00ffff;
    transition: 0.3s;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #0066ff, #00ffff);
    box-shadow: 0px 0px 25px #00ffff;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# ==========================
# App Title
# ==========================
st.title("🎓 Student Exam Score Predictor")
st.write("Upload any dataset and I’ll clean it + detect the exam score column automatically 🚀")

# ==========================
# Upload Data
# ==========================
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.write("📊 Preview of your data:", df.head())

    # --------------------------
    # AUTO DATA CLEANING
    # --------------------------
    # Drop useless columns (like IDs, names)
    drop_cols = [col for col in df.columns if "id" in col.lower() or "name" in col.lower()]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Drop completely empty columns
    df.dropna(axis=1, how="all", inplace=True)

    # Fill missing values
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Detect exam score column
    possible_targets = [col for col in df.columns if any(word in col.lower() for word in ["exam", "score", "marks", "result", "grade"])]
    target_col = None
    if len(possible_targets) == 1:
        target_col = possible_targets[0]
        st.info(f"🕵️ Detected exam score column: **{target_col}**")
    elif len(possible_targets) > 1:
        target_col = st.selectbox("⚡ Multiple possible target columns found. Please choose:", possible_targets)
    else:
        target_col = st.selectbox("❓ Couldn’t detect automatically. Please choose exam score column:", df.columns)

    # Features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode text columns
    encoders = {}
    for col in X.columns:
        if X[col].dtype == "object":
            encoders[col] = LabelEncoder()
            X[col] = encoders[col].fit_transform(X[col].astype(str))

    # Train model
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    st.success("🤖 Model trained successfully!")

    # ==========================
    # User-friendly Input Form
    # ==========================
    st.subheader("📝 Enter Student Information")

    input_data = {}
    for col in X.columns:
        if col in encoders:  # categorical
            options = encoders[col].classes_
            choice = st.selectbox(f"{col}", options)
            input_data[col] = encoders[col].transform([choice])[0]
        else:  # numeric
            min_val = int(df[col].min())
            max_val = int(df[col].max())
            val = st.slider(f"{col}", min_val, max_val, int(df[col].mean()))
            input_data[col] = val

    # ==========================
    # Predict
    # ==========================
    if st.button("✨ Predict Exam Score"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.success(f"📘 Predicted Exam Score: **{prediction:.2f}** / 100")
