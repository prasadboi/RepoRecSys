"""
Streamlit frontend for GitHub Repository Recommendations
"""
import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="GitHub Repository Recommender",
    page_icon="⭐",
    layout="wide"
)

st.title("⭐ GitHub Repository Recommendation System")
st.markdown("Get personalized repository recommendations based on your preferences!")

# API endpoint - change this to your deployed URL
API_URL = st.sidebar.text_input(
    "API URL",
    value="http://localhost:8000",  # Will use VM's localhost since running on VM
    help="API URL (use localhost:8000 when running on same VM)"
)

def get_recommendations(user_id: int, top_k: int = 10):
    """Call the recommendation API"""
    try:
        response = requests.post(
            f"{API_URL}/recommend",
            json={"user_id": user_id, "top_k": top_k},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling API: {str(e)}")
        return None

# Sidebar
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Number of recommendations", 5, 50, 10)

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Input")
    user_id = st.number_input(
        "User ID",
        min_value=1,
        value=1,
        help="Enter a user ID from the training dataset"
    )
    
    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Fetching recommendations..."):
            result = get_recommendations(user_id, top_k)
            if result:
                st.session_state['recommendations'] = result

with col2:
    st.header("Recommendations")
    
    if 'recommendations' in st.session_state:
        recommendations = st.session_state['recommendations']['recommendations']
        
        if recommendations:
            st.success(f"Found {len(recommendations)} recommendations for User {user_id}")
            
            # Display as table
            df_data = []
            for i, rec in enumerate(recommendations, 1):
                df_data.append({
                    'Rank': i,
                    'Project ID': rec['project_id'],
                    'Score': f"{rec['score']:.4f}",
                    'Language': rec.get('language', 'N/A'),
                    'Watchers': rec.get('watchers', 0),
                    'GitHub URL': rec['github_url']
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
            
            # Display as cards
            st.subheader("Repository Details")
            for i, rec in enumerate(recommendations, 1):
                with st.expander(f"#{i} - Project {rec['project_id']} (Score: {rec['score']:.4f})"):
                    st.markdown(f"**GitHub URL:** [{rec['github_url']}]({rec['github_url']})")
                    st.markdown(f"**Language:** {rec.get('language', 'N/A')}")
                    st.markdown(f"**Watchers:** {rec.get('watchers', 0)}")
                    st.markdown(f"**Recommendation Score:** {rec['score']:.4f}")
        else:
            st.warning("No recommendations found")
    else:
        st.info("Enter a User ID and click 'Get Recommendations' to start")

# Footer
st.markdown("---")
st.markdown("**GitHub Repository Recommendation System** - Powered by Two-Tower Neural Collaborative Filtering")

