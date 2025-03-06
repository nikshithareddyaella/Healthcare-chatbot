# Healthcare Chatbot (Version 1)

## Overview
This is the first version of the *Healthcare Chatbot, designed to assist users in identifying symptoms, diagnosing potential diseases, and providing recommendations. It uses a **rule-based approach* and pre-defined datasets for symptom analysis. The chatbot leverages *Decision Tree Classifier* and *Support Vector Machine (SVM)* for disease prediction.

---

## Features
- *Symptom Analysis*: Analyzes user-provided symptoms to identify potential diseases.
- *Disease Description*: Provides descriptions of identified diseases.
- *Precautionary Measures*: Suggests precautions for managing symptoms.
- *Severity Assessment*: Evaluates the severity of symptoms.
- *Interactive Chat Interface*: Asks users for symptoms and provides predictions, descriptions, and precautions.

---

## Tech Stack
- *Programming Language*: Python
- *Libraries*: Pandas, Scikit-learn, CSV
- *Models*: Decision Tree Classifier, Support Vector Machine (SVM)

---

## Files
- *chat_bot.py*: Main script for the chatbot.
- *Training.csv*: Dataset for training the model.
- *Testing.csv*: Dataset for testing the model.
- *symptom_Description.csv*: Descriptions of diseases.
- *symptom_precaution.csv*: Precautions for symptoms.
- *symptom_severity.csv*: Severity levels of symptoms.

---

## Setup Instructions

### 1. Clone the Repository
bash
git clone https://github.com/K-Tarunkumar/Healthcare-chatbot.git
cd Healthcare-chatbot/chatbot\ version1


### 2. Install Dependencies
Ensure you have Python installed. Then, install the required libraries:
bash
pip install pandas scikit-learn


### 3. Run the Chatbot
bash
python chat_bot.py


---

## Usage
1. Run the chat_bot.py script.
2. Enter your symptoms when prompted.
3. The chatbot will:
   - Identify potential diseases.
   - Provide descriptions and precautions.
   - Assess symptom severity.

---

## Example Interaction

Enter the symptom you are experiencing: itching
Okay. From how many days?: 3
Are you experiencing skin_rash? (yes/no): yes
Are you experiencing nodal_skin_eruptions? (yes/no): no
You may have Fungal infection.
Description: Fungal infections are caused by fungi and can affect the skin, nails, and hair.
Take the following measures:
1) Keep the area clean
2) Apply antifungal cream
3) Consult a doctor
4) Avoid sharing personal items


---

## Future Improvements
- *Expand Dataset*: Add more symptoms and diseases for better accuracy.
- *User Interface*: Build a GUI using Tkinter, Streamlit, or Flask.
- *Text-to-Speech*: Use pyttsx3 to read out predictions and precautions.
- *Deploy*: Deploy the chatbot as a web application using FastAPI or Flask.

