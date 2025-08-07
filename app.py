import streamlit as st
import pandas as pd
import numpy as np
import pickle

with open('./data/bodyfat_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

top5_features = [
    'Density',
    'Knee',
    'Abdomen',
    'Weight',
    'Height'
]

st.title('Body Fat Estimator')
st.write('Enter your measurements below to estimate your body fat percentage using a trained machine learning model.')



# Descriptions and (optional) image URLs for each feature
feature_descriptions = {
    'Density': {
        'help': 'Body density is typically measured using hydrostatic weighing or air displacement plethysmography. If you do not know your density, consult a health professional.',
        'image': None
    },
    'Knee': {
        'help': 'Knee circumference: Measure around the largest part of your knee while standing.',
        'image': './images/knee_measurement.jpg'
    },
    'Abdomen': {
        'help': 'Abdomen circumference: Measure at the level of the navel (belly button) while standing, keeping the tape horizontal.',
        'image': './images/abdomen_measurement.jpg'
    },
    'Weight': {
        'help': 'Your body weight in kilograms. Use a reliable scale for best accuracy.',
        'image': None
    },
    'Height': {
        'help': 'Your height in centimeters. Stand straight against a wall without shoes for best accuracy.',
        'image': None
    }
}


# Use columns to place image beside the form input
user_input = {}
for feature in top5_features:
    desc = feature_descriptions[feature]
    if desc['image']:
        col1, col2 = st.columns([2,1])
        with col1:
            if feature == 'Density':
                user_input[feature] = st.number_input(
                    'Enter Density (g/cm³):', min_value=0.8, max_value=1.2, step=0.001, format='%.3f', help=desc['help']
                )
            elif feature == 'Weight':
                user_input[feature] = st.number_input(
                    'Enter Weight (kg):', min_value=0.0, step=0.1, help=desc['help']
                )
            else:
                input_container = col1.container()
                if feature in ['Knee', 'Abdomen']:
                    user_input[feature] = input_container.number_input(
                        f'Enter {feature} (cm):', min_value=0.0, step=0.1, help=desc['help'], key=f'{feature}_input',
                        label_visibility="visible",
                        # Streamlit does not have a direct width param for number_input, so use container style
                    )
                    input_container.markdown('<style>div[data-testid="stNumberInput"] input {width: 120px;}</style>', unsafe_allow_html=True)
                else:
                    user_input[feature] = st.number_input(
                        f'Enter {feature} (cm):', min_value=0.0, step=0.1, help=desc['help']
                    )
        with col2:
            st.image(desc['image'], caption=f"How to measure {feature}", width=120, use_container_width=False)
    else:
        if feature == 'Density':
            user_input[feature] = st.number_input(
                'Enter Density (g/cm³):', min_value=0.8, max_value=1.2, step=0.001, format='%.3f', help=desc['help']
            )
        elif feature == 'Weight':
            user_input[feature] = st.number_input(
                'Enter Weight (kg):', min_value=0.0, step=0.1, help=desc['help']
            )
        else:
            if feature in ['Knee', 'Abdomen']:
                input_container = st.container()
                user_input[feature] = input_container.number_input(
                    f'Enter {feature} (cm):', min_value=0.0, step=0.1, help=desc['help'], key=f'{feature}_input',
                    label_visibility="visible",
                )
                input_container.markdown('<style>div[data-testid="stNumberInput"] input {width: 120px;}</style>', unsafe_allow_html=True)
            else:
                user_input[feature] = st.number_input(
                    f'Enter {feature} (cm):', min_value=0.0, step=0.1, help=desc['help']
                )


all_filled = all(v != 0.0 for v in user_input.values())
predict_btn = st.button('Estimate Body Fat', disabled=not all_filled)
if not all_filled:
    st.info('Please enter all required values above to enable prediction.')
if predict_btn:
    try:
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        if prediction < 0 or prediction > 70:
            st.warning(f"Estimated Body Fat Percentage: {prediction:.2f}%. This value is outside typical human ranges. Please check your inputs.")
        else:
            st.success(f'Estimated Body Fat Percentage: {prediction:.2f}%')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown('---')
st.markdown('''
### About Body Fat Percentage

**Body fat percentage** is a measure of the proportion of fat in your body compared to everything else (muscle, bone, water, etc.). It is an important indicator of health and fitness. Maintaining a healthy body fat percentage can help reduce the risk of chronic diseases such as heart disease, diabetes, and hypertension.

#### Healthy Body Fat Ranges (American Council on Exercise):
- **Men:**
  - Essential fat: 2-5%
  - Athletes: 6-13%
  - Fitness: 14-17%
  - Average: 18-24%
  - Obese: 25% and higher
- **Women:**
  - Essential fat: 10-13%
  - Athletes: 14-20%
  - Fitness: 21-24%
  - Average: 25-31%
  - Obese: 32% and higher

**Note:** These ranges are general guidelines. Individual health and fitness goals may vary. For personalized advice, consult a healthcare or fitness professional.

This app uses a Random Forest model trained on the top 5 most important features. For best results, use accurate and recent measurements.
''')
