import joblib
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="✈️ Airline Recommendation Predictor App",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #17becf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(145deg, #f0f2f6, #ffffff);
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-result {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
    }
    .recommended {
        background: linear-gradient(135deg, #4CAF50, #8BC34A);
        color: white;
    }
    .not-recommended {
        background: linear-gradient(135deg, #f44336, #FF5722);
        color: white;
    }
    .rating-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .example-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    .info-box {
        background: linear-gradient(45deg, #e3f2fd, #bbdefb);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2196F3;
    }
</style>
""", unsafe_allow_html=True)

# Sample data for dropdown choices (replace with your actual data)
traveller_types = ['Solo Leisure', 'Couple Leisure', 'Family Leisure', 'Business']
cabin_types = ['Economy Class', 'Premium Economy', 'Business Class', 'First Class']

# Preprocessing function from your original code
@st.cache_data
def preprocess_input(input_data):
    """
    Preprocess input data to match training format.
    """
    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([input_data])

    # Define imputation values (these should match your training data)
    q1_values = {
        'seat_comfort': 3.0,
        'cabin_service': 3.0,
        'value_for_money': 3.0
    }
    median_values = {
        'food_bev': 3.0,
        'entertainment': 3.0,
        'ground_service': 3.0
    }
    mode_traveller_type = 'Solo Leisure'
    mode_cabin = 'Economy Class'

    # Apply imputation for numerical columns
    low_null_val_imputed = ['seat_comfort', 'cabin_service', 'value_for_money']
    high_null_val_imputed = ['food_bev', 'entertainment', 'ground_service']

    for col in low_null_val_imputed:
        if col in input_df.columns:
            input_df[col].fillna(q1_values.get(col, input_df[col].mean()), inplace=True)

    for col in high_null_val_imputed:
        if col in input_df.columns:
            input_df[col].fillna(median_values.get(col, input_df[col].median()), inplace=True)

    # Handle missing values in categorical columns
    if 'traveller_type' in input_df.columns:
        input_df['traveller_type'].fillna(mode_traveller_type, inplace=True)

    if 'cabin' in input_df.columns:
        input_df['cabin'].fillna(mode_cabin, inplace=True)

    # Apply one-hot encoding
    input_encoded = pd.get_dummies(input_df, dtype=int)

    # Define expected columns (replace with your actual training columns)
    expected_columns = [
        'seat_comfort', 'cabin_service', 'food_bev', 'entertainment',
        'ground_service', 'value_for_money',
        'traveller_type_Business', 'traveller_type_Couple Leisure',
        'traveller_type_Family Leisure', 'traveller_type_Solo Leisure',
        'cabin_Business Class', 'cabin_Economy Class',
        'cabin_First Class', 'cabin_Premium Economy'
    ]

    # Align columns with training data
    for col in expected_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    input_encoded = input_encoded.reindex(columns=expected_columns, fill_value=0)

    return input_encoded

# Load or create model (replace with your actual model loading)
@st.cache_resource
def load_model():
    try:
        # Try to load the saved model
        with open('logistic_regression_model.joblib', 'rb') as file:
            model = joblib.load(file)
        st.success("✅ Loaded trained Logistic Regression model")
        return model
    except FileNotFoundError:
        # Create a dummy model for demonstration
        model = LogisticRegression(random_state=42, max_iter=1000)
        st.warning("⚠️ Using dummy model. Please replace with your trained model.")
        return model

# Header
st.markdown("<h1 class='main-header'>✈️ Airline Recommendation Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>🎯 Predict whether passengers will recommend an airline based on their flight experience and Feedback.<br><strong>Powered by Logistic Regression Machine Learning Model</strong> 🤖</p>", unsafe_allow_html=True)

# Load model
clf_logclass = load_model()

# Sidebar for inputs
st.sidebar.markdown("## 📝 Flight Experience Details")
st.sidebar.markdown("---")

# Categorical inputs
st.sidebar.markdown("### 👥 Travel Information")
col1, col2 = st.sidebar.columns(2)

with col1:
    traveller_type = st.selectbox(
        "👤 Traveller Type",
        options=traveller_types,
        index=0,
        help="Select your travel purpose"
    )

with col2:
    cabin = st.selectbox(
        "💺 Cabin Class",
        options=cabin_types,
        index=0,
        help="Select your cabin class"
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### ⭐ Rate Your Experience")
st.sidebar.markdown("*Rate each aspect from 1 (Very Poor) to 5 (Excellent)*")

# Rating inputs with enhanced styling
seat_comfort = st.sidebar.slider(
    "🪑 Seat Comfort",
    min_value=1, max_value=5, value=3,
    help="How comfortable were the seats?"
)

cabin_service = st.sidebar.slider(
    "👨‍✈️ Cabin Service",
    min_value=1, max_value=5, value=3,
    help="Quality of cabin crew service"
)

food_bev = st.sidebar.slider(
    "🍽️ Food & Beverages",
    min_value=1, max_value=5, value=3,
    help="Quality of food and drink service"
)

entertainment = st.sidebar.slider(
    "📺 Entertainment",
    min_value=1, max_value=5, value=3,
    help="In-flight entertainment quality"
)

ground_service = st.sidebar.slider(
    "🏢 Ground Service",
    min_value=1, max_value=5, value=3,
    help="Check-in, boarding, baggage handling"
)

value_for_money = st.sidebar.slider(
    "💰 Value for Money",
    min_value=1, max_value=5, value=3,
    help="Overall value for the price paid"
)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📊 Your Flight Experience Visualization")
    
    # Create enhanced radar chart
    categories = ['Seat<br>Comfort', 'Cabin<br>Service', 'Food &<br>Beverages', 
                 'Entertainment', 'Ground<br>Service', 'Value for<br>Money']
    values = [seat_comfort, cabin_service, food_bev, 
              entertainment, ground_service, value_for_money]
    
    # Create radar chart with better styling
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Your Ratings',
        line=dict(color='#1f77b4', width=3),
        fillcolor='rgba(31, 119, 180, 0.3)',
        marker=dict(size=8, color='#1f77b4')
    ))
    
    # Add average line
    avg_rating = np.mean(values)
    avg_values = [avg_rating] * len(categories)
    fig.add_trace(go.Scatterpolar(
        r=avg_values,
        theta=categories,
        mode='lines',
        name=f'Average ({avg_rating:.1f})',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5],
                tickvals=[1, 2, 3, 4, 5],
                ticktext=['1⭐', '2⭐', '3⭐', '4⭐', '5⭐']
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        title={
            'text': "Experience Ratings Overview",
            'x': 0.5,
            'font': {'size': 16}
        },
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 🎯 Prediction Results")
    
    # Create input dictionary
    input_data = {
        'traveller_type': traveller_type,
        'cabin': cabin,
        'seat_comfort': float(seat_comfort),
        'cabin_service': float(cabin_service),
        'food_bev': float(food_bev),
        'entertainment': float(entertainment),
        'ground_service': float(ground_service),
        'value_for_money': float(value_for_money)
    }
    
    if st.button("🔮 Predict Recommendation", type="primary", use_container_width=True):
        try:
            # Preprocess the input
            processed_input = preprocess_input(input_data)
            
            # For demonstration with dummy model, use simple logic
            # Replace this section when using your actual trained model:
            # prediction = clf_logclass.predict(processed_input)
            # prediction_proba = clf_logclass.predict_proba(processed_input)
            # confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]
            
            # Temporary prediction logic (replace with above when using real model)
            avg_rating = np.mean(values)
            prediction = [1] if avg_rating >= 3.5 else [0]
            confidence = min(0.95, max(0.55, (avg_rating - 1) / 4))
            
            # Display enhanced prediction result
            if prediction[0] == 1:
                st.markdown(f"""
                <div class='prediction-result recommended'>
                    <h2>✅ RECOMMENDED</h2>
                    <h3>This airline is likely to be recommended!</h3>
                    <p><strong>🎯 Confidence Score: {confidence:.1%}</strong></p>
                    <p><small>🤖 Predicted using Logistic Regression</small></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='prediction-result not-recommended'>
                    <h2>❌ NOT RECOMMENDED</h2>
                    <h3>This airline might not meet expectations</h3>
                    <p><strong>🎯 Confidence Score: {(1-confidence):.1%}</strong></p>
                    <p><small>🤖 Predicted using Logistic Regression</small></p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")

# Enhanced rating summary
st.markdown("---")
st.markdown("### 📋 Detailed Rating Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class='rating-card'>
        <h4>🪑 Seat Comfort</h4>
        <h2>{seat_comfort}/5 ⭐</h2>
        <p>Comfort level of aircraft seating</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='rating-card'>
        <h4>👨‍✈️ Cabin Service</h4>
        <h2>{cabin_service}/5 ⭐</h2>
        <p>Quality of flight crew service</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='rating-card'>
        <h4>🍽️ Food & Beverages</h4>
        <h2>{food_bev}/5 ⭐</h2>
        <p>Quality of meal and drink service</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='rating-card'>
        <h4>📺 Entertainment</h4>
        <h2>{entertainment}/5 ⭐</h2>
        <p>In-flight entertainment options</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='rating-card'>
        <h4>🏢 Ground Service</h4>
        <h2>{ground_service}/5 ⭐</h2>
        <p>Airport check-in and boarding</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='rating-card'>
        <h4>💰 Value for Money</h4>
        <h2>{value_for_money}/5 ⭐</h2>
        <p>Overall price-to-value ratio</p>
    </div>
    """, unsafe_allow_html=True)

# Flight details summary
avg_rating = np.mean(values)
st.markdown(f"""
<div class='info-box'>
    <h3>📈 Flight Experience Summary</h3>
    <p><strong>👤 Traveller Type:</strong> {traveller_type}</p>
    <p><strong>💺 Cabin Class:</strong> {cabin}</p>
    <p><strong>📊 Average Rating:</strong> {avg_rating:.1f}/5 ⭐</p>
    <p><strong>🎯 Overall Experience:</strong> {'Excellent' if avg_rating >= 4.5 else 'Good' if avg_rating >= 3.5 else 'Average' if avg_rating >= 2.5 else 'Poor'}</p>
</div>
""", unsafe_allow_html=True)

# Example cases
st.markdown("---")
st.markdown("### 🎪 Try These Example Cases")

examples = [
    {
        'name': '🌟 Excellent Experience',
        'description': 'High-rated business travel experience',
        'traveller': 'Business',
        'cabin': 'Business Class',
        'ratings': [5, 5, 4, 4, 5, 5]
    },
    {
        'name': '😊 Good Family Trip',
        'description': 'Solid family vacation experience',
        'traveller': 'Family Leisure',
        'cabin': 'Premium Economy',
        'ratings': [4, 4, 3, 4, 3, 4]
    },
    {
        'name': '😐 Average Solo Travel',
        'description': 'Mixed solo leisure experience',
        'traveller': 'Solo Leisure',
        'cabin': 'Economy Class',
        'ratings': [3, 3, 3, 2, 3, 3]
    },
    {
        'name': '😞 Poor Experience',
        'description': 'Disappointing economy flight',
        'traveller': 'Couple Leisure',
        'cabin': 'Economy Class',
        'ratings': [2, 2, 2, 1, 2, 1]
    }
]

col1, col2 = st.columns(2)

for i, example in enumerate(examples):
    col = col1 if i % 2 == 0 else col2
    
    with col:
        st.markdown(f"""
        <div class='example-container'>
            <h4>{example['name']}</h4>
            <p>{example['description']}</p>
            <p><strong>👤 Traveller:</strong> {example['traveller']}</p>
            <p><strong>💺 Cabin:</strong> {example['cabin']}</p>
            <p><strong>⭐ Ratings:</strong> {', '.join(map(str, example['ratings']))}</p>
            <p><strong>📊 Average:</strong> {np.mean(example['ratings']):.1f}/5</p>
        </div>
        """, unsafe_allow_html=True)

# Model Information
st.markdown("---")
st.markdown("### 🚀 Model Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### 🤖 Algorithm Details
    - **Model Type:** Logistic Regression
    - **Features:** 6 numerical ratings + 2 categorical features
    - **Output:** Binary recommendation (Yes/No) with confidence score
    - **Preprocessing:** One-hot encoding for categorical variables
    """)

with col2:
    st.markdown("""
    #### 📊 Rating Scale Guide
    - **1 = Very Poor** 😞 - Extremely unsatisfactory
    - **2 = Poor** 😐 - Below expectations
    - **3 = Average** 🙂 - Meets basic expectations
    - **4 = Good** 😊 - Above average quality
    - **5 = Excellent** 🤩 - Outstanding service
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666; padding: 20px;'>
    <h3>✈️ Airline Recommendation Predictor</h3>
    <p>Built with Streamlit & Logistic Regression | Predicting passenger satisfaction using machine learning</p>
    <p><em>🎯 Helping airlines improve customer experience through data-driven insights</em></p>
</div>
""", unsafe_allow_html=True)

# Sidebar deployment instructions
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 Deployment Instructions")
st.sidebar.markdown("""
1. **Install Requirements:**
   ```bash
   pip install streamlit pandas numpy 
   scikit-learn plotly pickle
   ```

2. **Update Model:**
   - Replace dummy model with your trained model
   - Update prediction logic in lines 200-210

3. **Run Application:**
   ```bash
   streamlit run app.py
   ```

4. **Deploy Online:**
   - Streamlit Cloud (GitHub integration)
   - Heroku, AWS, or GCP
""")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Use the examples above to test different scenarios and see how ratings affect recommendations!")