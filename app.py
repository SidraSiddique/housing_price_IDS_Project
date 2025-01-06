import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit.components.v1 as components

# Custom CSS to style the sidebar and buttons
st.markdown("""
    <style>
    /* Change the sidebar background color to white */
    .css-1d391kg {
        background-color: black;  /* White background */
    }

    /* Style the buttons in the sidebar */
    .stButton > button {
        background-color: white;  /* Navy Blue */
        color: #000080;  /* White Text */
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
        width: 100%;  /* Full width button */
    }

    .stButton > button:hover {
        background-color: #000066;  /* Darker Navy Blue for hover effect */
    }

    /* Change the color of the sidebar text to black */
    .css-1p7z1f8 {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


# Set background image
background_image_path = r"C:\Users\PMLS\Desktop\Housing-Prices-Project\images\house.jpg"
background_html = """
    <style>
    body {
        background-image: url("{background_image_path}");  /* Relative path to the image */
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #000000;  /* Ensure text color is black */
    }

    #main {
        background: rgba(255, 255, 255, 0.8);  /* White background with 80% opacity for content area */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);  /* Adding a slight shadow to make the content area stand out */
    }

    h1, h2, h3, h4, p {
        color: #003366;  /* Set text color for headings and paragraphs */
    }
    </style>
    """
# Inject custom HTML to apply the background image
components.html(background_html, height=0)

# Load the dataset
df = pd.read_csv('housing_prices.csv')

# Custom Title and Styling
st.markdown("<h1 style='text-align: center; font-size: 40px; font-weight: bold;'>Housing Price Prediction Analysis</h1>", unsafe_allow_html=True)

# Add a main container for content to ensure background overlay
st.markdown('<div id="main">', unsafe_allow_html=True)

# Create a sidebar for buttons and interactions

button_overview = st.sidebar.button("Dataset Overview")
button_summary = st.sidebar.button("Summary Statistics")

# Main Visualization Section with Nested Buttons
visualization_option = st.sidebar.radio("Choose Visualization", ["None", "Visualizations"])

if visualization_option == "Visualizations":
    # Create a set of nested buttons inside this "Visualizations" category
    visualization_choice = st.sidebar.radio("Choose a Visualization", [
        "Price Distribution (Histogram)",
        "Missing Values Heatmap",
        "Correlation Heatmap",
        "Pairplot of Selected Features",
        "Price Distribution by Bedrooms",
        "Price vs Area Scatter Plot",
        "Price by Parking Availability"
    ])

    # Displaying corresponding visual based on user selection
    if visualization_choice == "Price Distribution (Histogram)":
        st.write("<h4 style='font-size: 20px; font-weight: bold;'>Price Distribution (Histogram with KDE)</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df['price'], kde=True, bins=30, ax=ax, color='skyblue')
        st.pyplot(fig)

    elif visualization_choice == "Missing Values Heatmap":
        st.write("<h4 style='font-size: 20px; font-weight: bold;'>Missing Values Heatmap</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
        st.pyplot(fig)

    elif visualization_choice == "Correlation Heatmap":
        categorical_cols = df.select_dtypes(include=['object']).columns

    # Use .map() to convert categorical columns like 'yes'/'no' to numeric values
        for col in categorical_cols:
            if df[col].dtype == 'object':
                df[col] = df[col].map({'yes': 1, 'no': 0}).fillna(df[col])

        # Perform one-hot encoding for any remaining categorical columns
        df = pd.get_dummies(df, drop_first=True)

        # Now calculate and plot the correlation heatmap
        st.write("<h4 style='font-size: 20px; font-weight: bold;'>Correlation Heatmap</h4>", unsafe_allow_html=True)

        # Generate the heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title("Feature Correlation", fontsize=22, fontweight='bold')

        # Display the plot in Streamlit
        st.pyplot(fig)

    elif visualization_choice == "Pairplot of Selected Features":
        st.write("<h4 style='font-size: 20px; font-weight: bold;'>Pairplot of Selected Features</h4>", unsafe_allow_html=True)
        pairplot_fig = sns.pairplot(df[['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']], palette='coolwarm')
        st.pyplot(pairplot_fig)

    elif visualization_choice == "Price Distribution by Bedrooms":
        st.write("<h4 style='font-size: 20px; font-weight: bold;'>Price Distribution by Number of Bedrooms</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x=df['bedrooms'], y=df['price'], ax=ax, palette='Set2')
        ax.set_title("Price Distribution by Bedrooms", fontsize=22, fontweight='bold')
        st.pyplot(fig)

    elif visualization_choice == "Price vs Area Scatter Plot":
        st.write("<h4 style='font-size: 20px; font-weight: bold;'>Price vs Area</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=df['area'], y=df['price'], ax=ax)
        ax.set_title("Price vs Area", fontsize=22, fontweight='bold')
        ax.set_xlabel("Area", fontsize=18)
        ax.set_ylabel("Price", fontsize=18)
        st.pyplot(fig)

    elif visualization_choice == "Price by Parking Availability":
        st.write("<h4 style='font-size: 20px; font-weight: bold;'>Price Distribution by Parking Availability</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x=df['parking'], y=df['price'], ax=ax)
        ax.set_title("Price Distribution by Parking", fontsize=22, fontweight='bold')
        st.pyplot(fig)

# Display Dataset Overview Section
if button_overview:
    st.write("<h4 style='font-size: 20px; font-weight: bold;'>Dataset Overview</h4>", unsafe_allow_html=True)
    st.dataframe(df.head(10))  # Display first few rows of the dataset

# Display Summary Statistics Section
if button_summary:
    st.write("<h4 style='font-size: 20px; font-weight: bold;'>Summary Statistics</h4>", unsafe_allow_html=True)
    st.write(df.describe())  # Summary statistics for numerical features

# Model Evaluation Section (this section is unchanged)
button_model = st.sidebar.button("Model Evaluation")

if button_model:
    # Perform one-hot encoding for categorical columns (convert to dummy variables)
    df = pd.get_dummies(df, drop_first=True)

    # Ensure all columns are numeric (for scaling and machine learning)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Prepare features and target
    X = df.drop('price', axis=1)  # Features
    y = df['price']  # Target

    # Scaling the features using StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(X)

    # Train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(scaled_features, y)

    # Feature Importance
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1]
    sorted_features = X.columns[sorted_idx]
    sorted_importance = feature_importance[sorted_idx]

    # Feature Importance Plot
    st.write("<h4 style='font-size: 20px; font-weight: bold;'>Feature Importance</h4>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=sorted_importance, y=sorted_features, ax=ax, palette='Blues')
    plt.title('Feature Importance from Random Forest Model', fontsize=22, fontweight='bold')
    st.pyplot(fig)

    # Train-Test Split Evaluation
    st.write("<h4 style='font-size: 20px; font-weight: bold;'>Train-Test Split Evaluation</h4>", unsafe_allow_html=True)

    # Train-test split, Random Forest Model Evaluation (MAE, MSE, R^2)
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Model Evaluation: Displaying performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f'Mean Absolute Error: {mae:.2f}')
    st.write(f'Mean Squared Error: {mse:.2f}')
    st.write(f'R-squared: {r2:.2f}')
button_conclusion = st.sidebar.button("Conclusion")
# Conclusion Section
if button_conclusion:
    st.markdown("<h2 style='text-align: center; font-size: 32px; color: white;'>Conclusion: Key Takeaways from the Housing Price Prediction Project</h2>", unsafe_allow_html=True)

    # Displaying the data under key takeaways in a more structured format
    st.markdown("<h4 style='font-size: 24px; font-weight: bold; color: white;'>Data Exploration & Preprocessing:</h4>", unsafe_allow_html=True)
    st.markdown("""
    - The dataset provided valuable insights into the features influencing housing prices.
    - Key attributes such as area, number of bedrooms, bathrooms, and stories were found to be significant predictors of house prices.
    - Preprocessing steps like handling missing values and encoding categorical variables (e.g., converting "yes" and "no" to numeric values) were crucial in making the data suitable for modeling.
    - We also performed feature scaling to normalize the numerical data, ensuring the model performed optimally.
    """, unsafe_allow_html=True)

    st.markdown("<h4 style='font-size: 24px; font-weight: bold; color: white;'>Visualizations:</h4>", unsafe_allow_html=True)
    st.markdown("""
    - **Price Distribution**: The price distribution of the houses exhibited some skewness, which is common in real estate data where luxury properties can significantly affect price ranges.
    - **Correlation Heatmap**: The correlation heatmap highlighted strong relationships between certain variables, particularly area, bedrooms, and parking, with house prices.
    - **Pairplots and Boxplots**: Visualizing data through pairplots and boxplots provided further insights into how various features such as bedrooms and area relate to house prices, helping identify potential outliers and relationships.
    """, unsafe_allow_html=True)

    st.markdown("<h4 style='font-size: 24px; font-weight: bold; color: white;'>Modeling & Evaluation:</h4>", unsafe_allow_html=True)
    st.markdown("""
    - **Random Forest Regressor**: The model was trained using the Random Forest algorithm, which proved effective in predicting housing prices by learning from complex patterns in the data. Random Forest handles high-dimensional data well and provides useful feature importance insights.
    - **Evaluation Metrics**: The performance metrics (MAE, MSE, R²) confirmed that the model can predict house prices with a reasonable degree of accuracy. The R² value indicated a good fit, though improvements could be made by fine-tuning the model or experimenting with different algorithms.
    """, unsafe_allow_html=True)

    st.markdown("<h4 style='font-size: 24px; font-weight: bold; color: white;'>Feature Importance:</h4>", unsafe_allow_html=True)
    st.markdown("""
    - The area of the house, the number of bedrooms, and parking availability emerged as the most influential factors driving house prices, according to the feature importance plot.
    - This highlights the importance of spacious properties and key amenities in determining market value.
    """, unsafe_allow_html=True)

    st.markdown("<h4 style='font-size: 24px; font-weight: bold; color: white;'>Potential Improvements:</h4>", unsafe_allow_html=True)
    st.markdown("""
    - Further model improvement could involve tuning hyperparameters for the Random Forest model or exploring other algorithms like Gradient Boosting or XGBoost.
    - Additional feature engineering, such as including external factors like location, proximity to amenities, or even economic indicators, could improve prediction accuracy.
    """, unsafe_allow_html=True)
# Close main container
st.markdown('</div>', unsafe_allow_html=True)
