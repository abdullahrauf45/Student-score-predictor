import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================
# Futuristic CSS Styling (UNCHANGED)
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
    drop_cols = [col for col in df.columns if "id" in col.lower() or "name" in col.lower()]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")
    df.dropna(axis=1, how="all", inplace=True)

    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Detect exam score column
    possible_targets = [col for col in df.columns if any(word in col.lower() for word in ["exam", "score", "marks", "result", "grade"])]
    if len(possible_targets) == 1:
        target_col = possible_targets[0]
        st.info(f"🕵️ Detected exam score column: **{target_col}**")
    elif len(possible_targets) > 1:
        target_col = st.selectbox("⚡ Multiple possible target columns found. Please choose:", possible_targets)
    else:
        target_col = st.selectbox("❓ Couldn’t detect automatically. Please choose exam score column:", df.columns)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Identify categorical and numeric features
    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numeric_cols = [c for c in X.columns if X[c].dtype in ["int64", "float64"]]

    # Preprocessor: OneHotEncode categorical + Scale numeric
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    # Pipeline: preprocessing + regression
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.success("🤖 Model trained successfully!")
    st.write(f"📊 **R² Score:** {r2:.3f}")
    st.write(f"📊 **Mean Absolute Error (MAE):** {mae:.2f}")

    # ==========================
    # User-friendly Input Form
    # ==========================
    st.subheader("📝 Enter Student Information")

    input_data = {}
    for col in X.columns:
        if col in categorical_cols:
            choice = st.selectbox(f"{col}", df[col].unique())
            input_data[col] = choice
        else:
            min_val = int(df[col].min())
            max_val = int(df[col].max())
            val = st.slider(f"{col}", min_val, max_val, int(df[col].mean()))
            input_data[col] = val

    if st.button("✨ Predict Exam Score"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        max_score = df[target_col].max()
        st.success(f"📘 Predicted Exam Score: **{prediction:.2f}** / {max_score}")
