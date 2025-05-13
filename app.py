import streamlit as st
import numpy as np 
import pandas as pd 
import plotly.express as px
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Page config
st.set_page_config(
    page_title = 'Final Project in CCS230:Data Mining',
    layout = 'wide',
    initial_sidebar_state='expanded'
)

# Title 
st.title("Student Employability Prediction Using User Input Values via Slider 🤓 (Catboost Classifier)")

padtop = '<div style="padding: 20px; "></div>'  # added simple padding
st.markdown(padtop, unsafe_allow_html=True)

# Load dataset
df = pd.read_excel('Student-Employability-Datasets.xlsx')

df['Class_Encoded'] = LabelEncoder().fit_transform(df['CLASS'])

st.sidebar.subheader('Input features')

# Feature input sliders with a range of 1.0 to 5.0
general_appearance = st.sidebar.slider('General Appearance', 1, 5, 3, step=1)
manner_of_speaking = st.sidebar.slider('Manner of Speaking', 1, 5, 3, step=1)
physical_condition = st.sidebar.slider('Physical Condition', 1, 5, 3, step=1)
mental_alertness = st.sidebar.slider('Mental Alertness', 1, 5, 3, step=1)
self_confidence = st.sidebar.slider('Self-Confidence', 1, 5, 3, step=1)
ability_to_present_ideas = st.sidebar.slider('Ability to Present Ideas', 1, 5, 3, step=1)
communication_skills = st.sidebar.slider('Communication Skills', 1, 5, 3, step=1)
student_performance_rating = st.sidebar.slider('Student Performance Rating', 1, 5, 3, step=1)

# Features and targets
features = ['GENERAL APPEARANCE', 'MANNER OF SPEAKING',
       'PHYSICAL CONDITION', 'MENTAL ALERTNESS', 'SELF-CONFIDENCE',
       'ABILITY TO PRESENT IDEAS', 'COMMUNICATION SKILLS',
       'Student Performance Rating']

# Calculate EMPLOYABILITY_SCORE as the average of the selected features
df['EMPLOYABILITY_SCORE'] = df[features].mean(axis=1)

# Include EMPLOYABILITY_SCORE in the feature set
X = df[features + ['EMPLOYABILITY_SCORE']]
y = df['CLASS']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train classifier
catboost_model = CatBoostClassifier(
    iterations=500, 
    learning_rate=0.01, 
    depth=6, 
    verbose=False, 
    random_state=42, 
    bagging_temperature=2, 
    l2_leaf_reg=7
)
catboost_model.fit(X_train, y_train)

# User inputs
inputs = [
    general_appearance, 
    manner_of_speaking, 
    physical_condition, 
    mental_alertness, 
    self_confidence, 
    ability_to_present_ideas, 
    communication_skills, 
    student_performance_rating
]

# Calculate user input EMPLOYABILITY_SCORE
employability_score = np.mean(inputs)
inputs.append(employability_score)

# Prediction
y_pred = catboost_model.predict([inputs])
y_pred1 = catboost_model.predict(X_test)


input_feature = pd.DataFrame([inputs], columns=features + ['Employability Score'])

# Feature Importance Plot
st.subheader("Feature Importance Plot")
importances = catboost_model.feature_importances_
feature_names = X_train.columns

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names, palette='icefire', ax=ax)
ax.set_title('Feature Importance Plot of the CatBoost Model', fontsize=19)
ax.set_xlabel('Importance', fontsize=17)
ax.set_ylabel('Features', fontsize=17)
st.pyplot(fig)
st.write("""
The feature importance plot displays the relative impact of each input feature on the model’s predictions for student employability. The length of each bar indicates the strength of the contribution of the respective feature to the prediction outcome. Notably, the top influential features are:

- **Mental Alertness** – This feature has the highest importance score, indicating it plays a critical role in predicting employability based on the model.
- **Self-Confidence** and **Ability to Present Ideas** – These features also exhibit substantial importance, underscoring the relevance of effective communication and confidence in employability assessments.
- **Employability Score** – Interestingly, the combined average score also ranks as a significant predictor, suggesting that aggregating the individual features provides valuable insights into overall employability potential.

The remaining features still contribute to the model but to a lesser extent, suggesting that focusing on the top-performing features may offer targeted areas for skill development to improve employability outcomes.
""")

# Classification Report
st.subheader("Classification Report")
y_test_pred = catboost_model.predict(X_test)
report = classification_report(y_test, y_test_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Plotting Classification Report
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="YlOrBr", cbar=False, linewidths=0.5, ax=ax)
ax.set_title("Classification Report - CatBoost Model")
ax.set_xlabel("Metrics")
ax.set_ylabel("Classes")
st.pyplot(fig)

# Display user inputs
st.markdown(padtop, unsafe_allow_html=True)
st.markdown(padtop, unsafe_allow_html=True)
st.subheader('Input features based on slider:')
st.write(input_feature)

# Display results
st.markdown(padtop, unsafe_allow_html=True)
st.markdown(padtop, unsafe_allow_html=True)
st.subheader("Predictions 🎯")
st.metric('Result:', y_pred[0], border=True)

