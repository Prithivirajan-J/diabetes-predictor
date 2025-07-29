🩺 Diabetes Risk Predictor

A simple web application built with **Flask** and **Machine Learning** to predict the risk of diabetes based on health parameters. It uses a trained classification model on the **Pima Indians Diabetes dataset**.

---

## 📊 Features

- User-friendly form built with **Bootstrap**
- Predicts diabetes risk with probability score
- Displays personalized **risk comparison chart**
- Clean and responsive UI
- Pretrained model using **Random Forest Classifier**


### 1. Clone the Repository

git clone https://github.com/Prithivirajan-J/diabetes-predictor


2. Install Dependencies:

Create a virtual environment (recommended):

python -m venv venv

source venv/bin/activate     # On Windows: venv\Scripts\activate

Install required packages:

pip install -r requirements.txt

3. To Train the Model 

python train_model.py

4. Run the Flask App

python app.py

Visit: http://127.0.0.1:5000 in your browser.

🔧 Tools & Tech

-Python 3.11+

-Flask

-Scikit-learn

-Matplotlib

-Bootstrap 5
