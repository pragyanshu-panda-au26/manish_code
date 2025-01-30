import streamlit as st
import pandas as pd
import os
from recommender import HotelRecommender

st.set_page_config(page_title="Hotel Recommender", page_icon="🏨")

@st.cache_resource
def load_recommender():
    return HotelRecommender()

def format_confidence(confidence):
    """Format confidence score as percentage"""
    return f"{confidence * 100:.1f}%"

def main():
    st.title("🏨 Hotel Recommendation System")
    st.write("Get personalized hotel recommendations based on your user code!")
    
    recommender = load_recommender()
    
    # Load users for validation
    current_dir = os.path.dirname(os.path.abspath(__file__))
    users_df = pd.read_excel(os.path.join(current_dir, 'users.xlsx'))
    valid_user_codes = users_df['userCode'].unique()
    
    # User input
    user_code = st.text_input("Enter your User Code:")
    
    if user_code:
        try:
            user_code = int(user_code)
            if user_code in valid_user_codes:
                recommendations = recommender.get_recommendations(user_code)
                
                if recommendations:
                    st.subheader("Top 5 Recommended Hotels for You:")
                    
                    # Create a nice looking table for recommendations
                    for i, (hotel, confidence, reason) in enumerate(recommendations, 1):
                        with st.container():
                            col1, col2 = st.columns([3, 2])
                            with col1:
                                st.write(f"**{i}. {hotel}**")
                                st.write(f"_{reason}_")
                            with col2:
                                st.write(f"Confidence: **{format_confidence(confidence)}**")
                            st.divider()
                else:
                    st.warning("No recommendations found for this user.")
            else:
                st.error("Invalid user code. Please enter a valid user code.")
        except ValueError:
            st.error("Please enter a valid numeric user code.")

if __name__ == "__main__":
    main()
