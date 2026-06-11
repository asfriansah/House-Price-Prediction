

# 🏠 House Price Predictor & Market Analytics Dashboard

Ever wondered what dictates the wild price swings in the real estate market? This project is a **Streamlit-powered web application** that brings machine learning right to your fingertips. It doesn't just guess numbers; it analyzes patterns to provide smart, data-driven property valuations!

Built on top of the famous Advanced House Prices dataset, this dashboard uses a tuned **Ridge Regression** model under the hood to predict house prices based on key architectural and environmental features.

---

## ✨ Features

* **Instant Valuations:** Punch in your house metrics (Area, Quality, Age, Location) and get a price estimate in a split second.
* **Dynamic Data Recovery:** Zero friction with dirty data. The app auto-cleans missing values and converts datatypes seamlessly on the fly.
* **Live Market Insights:** Interactive data visualizations that break down pricing distribution by neighborhood and property quality.
* **Robust Preprocessing:** Handles complex One-Hot Encoding and feature scaling smoothly behind the scenes.

---

## 🛠️ The Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/) (Turning Python scripts into beautiful web apps)
* **Machine Learning:** `scikit-learn` (Ridge Regression & StandardScaler)
* **Data Wrangling:** `pandas` & `numpy`
* **Dataviz:** `matplotlib` & `seaborn`

---

## 🚀 Quick Start (Local Setup)

Want to run this dashboard on your own machine? It’s as easy as counting 1-2-3!

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
```
### 2. Install Dependencies
Make sure you have Python installed, then run:

```bash
pip install streamlit joblib pandas numpy matplotlib seaborn scikit-learn
```

### 3. Run the App
Fire up the local Streamlit server with:

```bash
streamlit run app.py
```
Your browser should automatically pop up showing the dashboard at `http://localhost:8501!`

---

## 📂 Project Structure

```bash
Plaintext
├── app.py                             # The main Streamlit dashboard application code
├── best_house_price_model_ridge.joblib # Trained Ridge Regression weights
├── train.csv                          # Real estate historical dataset used for scaling & live analytics
└── README.md                          # You are here!
```

---

💡 How It Works Under the Hood
1. The Blueprint: When the app boots up, it reads `train.csv` and dynamically isolates numeric and categorical features using native Pandas logic to avoid data-type mismatch errors.
2. The Alignment: Your manual inputs from the UI sliders and dropdowns are captured and transformed into exactly 259 features to match the strict mathematical shape expected by the Ridge model. Any unexpected columns are safely filtered out.
3. The Target: The model predicts the price in a logarithmic scale (to handle market skewness) and instantly reverses it using $e^x - 1$ `(np.expm1)` to show you the real-world dollar value.

---

🤝 Contributing
Got an idea to make the predictions even sharper? Or want to add a new tab for retraining the model dynamically?
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

📝 License
Distributed under the MIT License. See `LICENSE` for more information.  
Happy coding! Built with 💻 and ☕ by andri asfriansah.
