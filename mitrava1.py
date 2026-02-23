"""
MITRAVA – Goal Alignment & Digital Wellbeing Assistant
A Streamlit web app prototype for student project

Features:
- Mock user authentication
- Synthetic usage data generation
- Behavioral pattern clustering (K-Means)
- Goal alignment scoring
- Anonymous peer benchmarking
- Context switch insights
- Rule-based nudges
- Chatbot interface
- Admin panel
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import os
# ============================================================
# SECTION 1: PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="MITRAVA - Digital Wellbeing Assistant",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SECTION 2: MOCK USER DATABASE
# ============================================================
USERS_DB = {
    "alice": {"password": "alice123", "name": "Alice", "user_id": 1},
    "bob": {"password": "bob123", "name": "Bob", "user_id": 2},
    "charlie": {"password": "charlie123", "name": "Charlie", "user_id": 3},
    "diana": {"password": "diana123", "name": "Diana", "user_id": 4},
    "admin": {"password": "admin123", "name": "Admin", "user_id": 0, "is_admin": True}
}

# ============================================================
# SECTION 3: DATA GENERATION
# ============================================================
@st.cache_data
def generate_synthetic_data():
    """Generate realistic synthetic usage data for multiple users"""
    np.random.seed(42)
    random.seed(42)
    
    app_categories = {
        "Productivity": ["Microsoft Word", "Google Docs", "Notion", "Slack", "Zoom"],
        "Social": ["Instagram", "Twitter", "Facebook", "WhatsApp", "Snapchat"],
        "Entertainment": ["Netflix", "YouTube", "Spotify", "TikTok", "Games"],
        "Education": ["Coursera", "Khan Academy", "Duolingo", "Quizlet", "Study Notes"],
        "Utility": ["Calculator", "Calendar", "Weather", "Maps", "Settings"]
    }
    
    data = []
    user_ids = [1, 2, 3, 4]
    
    for user_id in user_ids:
        base_date = datetime.now() - timedelta(days=30)
        
        user_profile = random.choice(["studious", "social", "balanced", "entertained"])
        
        for day_offset in range(30):
            current_date = base_date + timedelta(days=day_offset)
            
            if user_profile == "studious":
                category_weights = {"Productivity": 0.3, "Social": 0.1, "Entertainment": 0.1, "Education": 0.4, "Utility": 0.1}
            elif user_profile == "social":
                category_weights = {"Productivity": 0.15, "Social": 0.4, "Entertainment": 0.25, "Education": 0.1, "Utility": 0.1}
            elif user_profile == "entertained":
                category_weights = {"Productivity": 0.1, "Social": 0.2, "Entertainment": 0.45, "Education": 0.15, "Utility": 0.1}
            else:
                category_weights = {"Productivity": 0.2, "Social": 0.2, "Entertainment": 0.2, "Education": 0.2, "Utility": 0.2}
            
            daily_sessions = random.randint(8, 20)
            
            for _ in range(daily_sessions):
                category = random.choices(
                    list(category_weights.keys()),
                    weights=list(category_weights.values())
                )[0]
                
                app_name = random.choice(app_categories[category])
                usage_minutes = max(5, int(np.random.exponential(30)))
                context_switch_count = random.randint(0, 5)
                
                data.append({
                    "user_id": user_id,
                    "date": current_date.strftime("%d-%m-%Y"),
                    "app_category": category,
                    "app_name": app_name,
                    "usage_minutes": usage_minutes,
                    "context_switch_count": context_switch_count
                })
    
    return pd.DataFrame(data)

# ============================================================
# SECTION 4: CLUSTERING (BEHAVIORAL PATTERNS)
# ============================================================
@st.cache_data
def perform_clustering(df):
    """Apply K-Means clustering based on usage patterns"""
    user_summary = df.groupby("user_id").apply(
        lambda x: pd.Series({
            "education_minutes": x[x["app_category"] == "Education"]["usage_minutes"].sum(),
            "social_minutes": x[x["app_category"] == "Social"]["usage_minutes"].sum(),
            "entertainment_minutes": x[x["app_category"] == "Entertainment"]["usage_minutes"].sum(),
            "productivity_minutes": x[x["app_category"] == "Productivity"]["usage_minutes"].sum(),
            "total_context_switches": x["context_switch_count"].sum()
        })
    ).reset_index()
    
    features = user_summary[["education_minutes", "social_minutes", "entertainment_minutes"]]
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    user_summary["cluster"] = kmeans.fit_predict(features)
    
    cluster_names = {0: "Focused Learner 📚", 1: "Social Butterfly 🦋", 2: "Entertainment Seeker 🎮"}
    user_summary["behavior_pattern"] = user_summary["cluster"].map(cluster_names)
    
    return user_summary, kmeans

# ============================================================
# SECTION 5: GOAL ALIGNMENT SCORING
# ============================================================
def calculate_goal_alignment(user_data, goal_mode):
    """
    Calculate goal alignment score based on usage and goal preference
    Goal modes: Focus (0), Balanced (1), Relaxed (2)
    """
    total_minutes = user_data["usage_minutes"].sum()
    
    if total_minutes == 0:
        return 50, "Partially Aligned"
    
    productivity = user_data[user_data["app_category"] == "Productivity"]["usage_minutes"].sum()
    education = user_data[user_data["app_category"] == "Education"]["usage_minutes"].sum()
    entertainment = user_data[user_data["app_category"] == "Entertainment"]["usage_minutes"].sum()
    social = user_data[user_data["app_category"] == "Social"]["usage_minutes"].sum()
    context_switches = user_data["context_switch_count"].sum()
    
    productive_ratio = (productivity + education) / total_minutes
    entertainment_ratio = entertainment / total_minutes
    social_ratio = social / total_minutes
    
    if goal_mode == 0:
        score = (productive_ratio * 60) + ((1 - entertainment_ratio) * 20) + (max(0, 20 - context_switches * 0.5))
    elif goal_mode == 1:
        balance_score = 100 - abs(productive_ratio - 0.4) * 100 - abs(entertainment_ratio - 0.3) * 100
        score = max(0, min(100, balance_score))
    else:
        score = (entertainment_ratio * 30) + (social_ratio * 30) + ((1 - productive_ratio) * 20) + 20
    
    score = max(0, min(100, score))
    
    if score >= 70:
        status = "Aligned ✅"
    elif score >= 40:
        status = "Partially Aligned ⚖️"
    else:
        status = "Drifting ⚠️"
    
    return score, status

# ============================================================
# SECTION 6: PEER BENCHMARKING
# ============================================================
def get_peer_percentile(user_id):
    """Generate simulated peer percentile for anonymous benchmarking"""
    random.seed(user_id * 42)
    return random.randint(1, 100)

def get_percentile_message(percentile):
    """Return appropriate message based on percentile"""
    if percentile < 40:
        return f"⚠️ You're in the bottom {percentile}% for productivity. Let's work on this together!", "warning"
    elif percentile <= 70:
        return f"📊 You're around the middle ({percentile}th percentile). Room for growth!", "info"
    else:
        return f"🌟 Amazing! You're in the top {100 - percentile}% for productivity!", "success"

# ============================================================
# SECTION 7: CONTEXT SWITCH ANALYSIS
# ============================================================
def generate_context_switch_details(user_id, selected_date):
    """Generate simulated context switch patterns based on user and date"""
    random.seed(user_id * 100 + selected_date.day + selected_date.month)
    return {
        "Study → Social": random.randint(5, 25),
        "Social → Entertainment": random.randint(3, 20),
        "Study → YouTube": random.randint(2, 15),
        "Work → Social": random.randint(4, 18),
        "Entertainment → Study": random.randint(1, 10)
    }

# ============================================================
# SECTION 8: NUDGE GENERATION
# ============================================================
def generate_nudges(user_data, percentile, goal_score, context_switches, goal_mode):
    """Generate personalized nudges based on user behavior"""
    nudges = []
    
    total_minutes = user_data["usage_minutes"].sum()
    entertainment = user_data[user_data["app_category"] == "Entertainment"]["usage_minutes"].sum()
    education = user_data[user_data["app_category"] == "Education"]["usage_minutes"].sum()
    social = user_data[user_data["app_category"] == "Social"]["usage_minutes"].sum()
    
    if context_switches > 50:
        nudges.append("🎯 Your app switching is too high—try a 20-min focus sprint!")
    
    if percentile < 40:
        nudges.append("💪 Your productivity is below average. Let's set a small goal together!")
    
    if total_minutes > 0 and entertainment / total_minutes > 0.4:
        nudges.append("🎬 Entertainment time is high today. Maybe balance with some learning!")
    
    if education < 30:
        nudges.append("📚 You haven't studied much today. Even 15 minutes can help!")
    
    if total_minutes > 0 and social / total_minutes > 0.3:
        nudges.append("💬 Try reducing social media after 10 PM, bro.")
    
    if goal_score < 40:
        nudges.append("🧭 You seem to be drifting from your goals. Take a moment to refocus!")
    
    if context_switches < 20:
        nudges.append("🌟 Low context switching detected – great focus today!")
    
    if education > entertainment:
        nudges.append("❤️ You studied more than you were entertained today. Proud of you!")
    
    if goal_mode == 0 and education > 60:
        nudges.append("🏆 Focus mode activated and you're crushing it!")
    
    if not nudges:
        nudges.append("✨ You're doing great! Keep up the balanced lifestyle!")
    
    return nudges[:4]

# ============================================================
# SECTION 9: CHATBOT LOGIC
# ============================================================
def get_chatbot_response(user_input, percentile, context_switches, goal_score):
    """Generate chatbot response based on user input and metrics"""
    user_input_lower = user_input.lower()
    
    if any(word in user_input_lower for word in ["tired", "exhausted", "sleepy"]):
        return "😴 Hey, sounds like you need a break! Step away from the screen, take a 10-minute walk, and come back refreshed. I'm here when you're ready! 💙"
    
    if any(word in user_input_lower for word in ["stressed", "anxious", "worried", "overwhelmed"]):
        return "🫂 I hear you. Stress happens to everyone. Try this: close your eyes, take 3 deep breaths. Remember, small steps lead to big wins. You've got this! 💪"
    
    if any(word in user_input_lower for word in ["unfocused", "distracted", "can't focus", "scattered"]):
        return "🎯 Focus feeling fuzzy? Try the 25-5 technique: 25 minutes of work, 5 minutes of rest. Start with just one small task. I believe in you! ✨"
    
    if any(word in user_input_lower for word in ["help", "what should i do", "advice"]):
        if percentile < 40:
            return f"📊 Looking at your patterns, you're in the {percentile}th percentile for productivity. Let's start small: pick ONE task and commit to 15 focused minutes. You can do it! 🚀"
        elif context_switches > 50:
            return "🔄 I noticed you're switching apps a lot. Try keeping just one app open at a time. It's like giving your brain a VIP experience! 🧠✨"
        else:
            return "🌟 You're actually doing pretty well! Keep maintaining your balance. Any specific area you want to work on?"
    
    if any(word in user_input_lower for word in ["hello", "hi", "hey"]):
        return "👋 Hey there! I'm MITRAVA, your digital wellbeing buddy. How are you feeling today? Tell me what's on your mind! 💭"
    
    if any(word in user_input_lower for word in ["thanks", "thank you", "appreciate"]):
        return "🤗 Always happy to help! Remember, taking care of your digital wellbeing is a journey, not a destination. I'm here whenever you need me! 💙"
    
    if goal_score < 40:
        return f"📈 I see your goal alignment is at {goal_score:.0f}%. No worries—every day is a fresh start! What's one small thing you can do right now to get back on track? 🎯"
    else:
            return " sorry but this is beyond my capabilities right now. However, I'm here to support you on your digital wellbeing journey! 💙"
    return "💭 I'm here to support your digital wellbeing journey! Tell me how you're feeling, or ask me for tips on focus, productivity, or taking breaks. 🌱"

# ============================================================
# SECTION 10: SESSION STATE INITIALIZATION
# ============================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================================
# SECTION 11: AUTHENTICATION UI
# ============================================================
def show_login_page():
    """Display login interface"""
    st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <h1>🧘 MITRAVA</h1>
        <h3>Goal Alignment & Digital Wellbeing Assistant</h3>
        <p style='color: gray;'>Your personal companion for balanced digital life</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔐 Login")
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                if username in USERS_DB and USERS_DB[username]["password"] == password:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.is_admin = USERS_DB[username].get("is_admin", False)
                    st.session_state.messages = []
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        st.divider()
        st.markdown("**⚠️ADMIN OR USER :WITHOUT PASSWORD YOU CAN'T ENTER!:**")
        st.markdown("    PASSWORD IS USERNAME FOLLOWED BY NUMBERS")
       

# ============================================================
# SECTION 12: ADMIN PANEL
# ============================================================
def show_admin_panel():
    """Display admin panel with full data access"""
    st.markdown("## 👨‍💼 Admin Panel")
    
    if st.sidebar.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.is_admin = False
        st.rerun()
    
    df = generate_synthetic_data()
    user_summary, kmeans = perform_clustering(df)
    
    tab1, tab2, tab3 = st.tabs(["📊 Full Dataset", "🔬 Clustering Analysis", "📖 Backend Logic"])
    
    with tab1:
        st.markdown("### Complete Usage Dataset")
        st.dataframe(df, use_container_width=True, height=400)
        
        st.markdown("### Dataset Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", len(df))
        col2.metric("Unique Users", df["user_id"].nunique())
        col3.metric("Date Range", f"{df['date'].min()} to {df['date'].max()}")
        col4.metric("Total Usage (min)", df["usage_minutes"].sum())
    
    with tab2:
        st.markdown("###  Clustering Results")
        st.markdown("**Clustering Features:** Education Minutes, Social Minutes, Entertainment Minutes")
        
        st.markdown("#### User Behavioral Patterns")
        st.dataframe(user_summary, use_container_width=True)
        
        st.markdown("#### Cluster Visualization")
        fig = px.scatter_3d(
            user_summary,
            x="education_minutes",
            y="social_minutes",
            z="entertainment_minutes",
            color="behavior_pattern",
            hover_data=["user_id", "productivity_minutes"],
            title="3D Cluster Visualization"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Cluster Centers")
        centers_df = pd.DataFrame(
            kmeans.cluster_centers_,
            columns=["Education", "Social", "Entertainment"]
        )
        centers_df["Cluster"] = [0, 1, 2]
        st.dataframe(centers_df, use_container_width=True)
    
    with tab3:
        st.markdown("### Backend Logic Documentation")
        
        st.markdown("""
        #### 1. Data Generation
        - Synthetic data is generated for 4 users over 30 days
        - Each user has a behavioral profile: studious, social, balanced, or entertained
        - Usage patterns are generated using exponential distribution for realistic session lengths
        
        #### 2. Clustering Algorithm
        - **Algorithm:** K-Means with 3 clusters
        - **Features:** Education minutes, Social minutes, Entertainment minutes
        - **Output:** Behavioral pattern labels (Focused Learner, Social Butterfly, Entertainment Seeker)
        
        #### 3. Goal Alignment Scoring
        - Score calculated based on:
          - Productivity/Education ratio
          - Entertainment ratio
          - Context switching frequency
        - Weights adjusted based on goal mode (Focus/Balanced/Relaxed)
        - Output categories: Aligned (≥70), Partially Aligned (40-69), Drifting (<40)
        
        #### 4. Peer Benchmarking
        - Percentile calculated using seeded random for consistency
        - Anonymous comparison without exposing individual data
        
        #### 5. Nudge Generation
        - Rule-based system checking:
          - High context switches (>50)
          - Low productivity percentile (<40)
          - High entertainment ratio (>40%)
          - Low education minutes (<30)
          - Goal alignment status
        
        #### 6. Chatbot Logic
        - Keyword-based response system
        - Integrates user metrics for personalized responses
        - Emotional, supportive tone
        """)

# ============================================================
# SECTION 13: USER DASHBOARD
# ============================================================
def show_user_dashboard():
    """Display user-specific dashboard"""
    username = st.session_state.username
    user_info = USERS_DB[username]
    user_id = user_info["user_id"]
    
    df = generate_synthetic_data()
    user_data = df[df["user_id"] == user_id].copy()
    user_data['date'] = pd.to_datetime(user_data['date'], format='%d-%m-%Y')
    user_summary, _ = perform_clustering(df)

    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👤 Welcome, {user_info['name']}!")

        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.is_admin = False
            st.rerun()

        st.divider()

        st.markdown("### 📅 Filters")
        available_dates_formatted = sorted(user_data["date"].dt.strftime("%d/%m/%Y").unique())
        selected_date_str = st.selectbox("Select Day", available_dates_formatted, index=len(available_dates_formatted)-1)
        selected_date = pd.to_datetime(selected_date_str, format="%d/%m/%Y")
        
        st.divider()
        
        st.markdown("### 🎯 Goal Preference")
        goal_mode = st.select_slider(
            "Your Focus Mode",
            options=[0, 1, 2],
            format_func=lambda x: ["🎯 Focus", "⚖️ Balanced", "😌 Relaxed"][x],
            value=1
        )
        
        st.markdown("""
        <small style='color: gray;'>
        - Focus: Prioritize productivity & learning<br>
        - Balanced: Equal mix of work & play<br>
        - Relaxed: More flexibility for entertainment
        </small>
        """, unsafe_allow_html=True)
    
    # Main content
    st.markdown("# 🧘 MITRAVA Dashboard")
    st.markdown("*Your personal digital wellbeing companion*")
    
    # Filter data for selected day
    daily_data = user_data[user_data["date"] == selected_date]
    
    # Get user's behavioral pattern
    user_pattern = user_summary[user_summary["user_id"] == user_id]["behavior_pattern"].values
    pattern_label = user_pattern[0] if len(user_pattern) > 0 else "Balanced User"
    
    # Calculate metrics
    daily_screen_time = daily_data["usage_minutes"].sum()
    weekly_data = user_data[user_data["date"] >= pd.to_datetime(selected_date) - timedelta(days=7)]
    weekly_screen_time = weekly_data["usage_minutes"].sum()
    total_context_switches = daily_data["context_switch_count"].sum()
    
    goal_score, goal_status = calculate_goal_alignment(daily_data, goal_mode)
    percentile = get_peer_percentile(user_id)
    
    st.divider()
    
    # ==================== USAGE OVERVIEW ====================
    st.markdown("## 📊 Usage Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📱 Daily Screen Time", f"{daily_screen_time} min")
    col2.metric("📅 Weekly Screen Time", f"{weekly_screen_time} min")
    col3.metric("🔄 Context Switches", total_context_switches)
    col4.metric("🧠 Your Pattern", pattern_label)
    
    st.divider()
    
    # Category breakdown & Weekly trend
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### 📈 Category-wise Usage")
        category_usage = daily_data.groupby("app_category")["usage_minutes"].sum().reset_index()
        
        bar_chart = alt.Chart(category_usage).mark_bar().encode(
            x=alt.X("app_category:N", title="Category", sort="-y"),
            y=alt.Y("usage_minutes:Q", title="Minutes"),
            color=alt.Color("app_category:N", legend=None, scale=alt.Scale(scheme="tableau10"))
        ).properties(height=300)
        
        st.altair_chart(bar_chart, use_container_width=True)
    
    with col_right:
        st.markdown("### 📅 Weekly Trend")
        weekly_trend = weekly_data.groupby("date")["usage_minutes"].sum().reset_index()
        
        line_chart = alt.Chart(weekly_trend).mark_line(point=True).encode(
            x=alt.X("date:T", title="Date", axis=alt.Axis(format="%d/%m/%Y")),
            y=alt.Y("usage_minutes:Q", title="Minutes"),
            tooltip=["date", "usage_minutes"]
        ).properties(height=300)
        
        st.altair_chart(line_chart, use_container_width=True)
    
    st.divider()
    

    # ==================== GOAL ALIGNMENT ====================
    st.markdown("## 🎯 Goal Alignment")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Alignment Score", f"{goal_score:.0f}/100")
        if "Aligned" in goal_status and "Partially" not in goal_status:
            st.success(goal_status)
        elif "Partially" in goal_status:
            st.info(goal_status)
        else:
            st.warning(goal_status)
    
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=goal_score,
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4CAF50" if goal_score >= 70 else "#FFC107" if goal_score >= 40 else "#F44336"},
                "steps": [
                    {"range": [0, 40], "color": "#FFEBEE"},
                    {"range": [40, 70], "color": "#FFF8E1"},
                    {"range": [70, 100], "color": "#E8F5E9"}
                ]
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # ==================== PEER BENCHMARKING ====================
    st.markdown("## 👥 Anonymous Peer Comparison")
    
    percentile_msg, msg_type = get_percentile_message(percentile)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Your Productivity Percentile", f"{percentile}th")
        if percentile >= 70:
            st.markdown(f"🏆 You're in the **top {100-percentile}%** for productivity!")
        elif percentile >= 40:
            st.markdown(f"📊 You're in the **middle {percentile}th** percentile")
        else:
            st.markdown(f"💪 You're in the **{percentile}th** percentile - room to grow!")
    
    with col2:
        if msg_type == "success":
            st.success(percentile_msg)
        elif msg_type == "info":
            st.info(percentile_msg)
        else:
            st.warning(percentile_msg)
    
    st.divider()
    
    # ==================== CONTEXT SWITCH INSIGHT ====================
    st.markdown("## 🔄 Context Switch Insights")
    
    st.info("💡 **Why it matters:** Frequent app switching fragments your attention and reduces focus. Studies show it takes ~23 minutes to regain deep focus after a switch!")
    
    switch_details = generate_context_switch_details(user_id, selected_date)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Switch Patterns")
        switch_df = pd.DataFrame([
            {"Transition": k, "Count": v} for k, v in switch_details.items()
        ])
        st.dataframe(switch_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### Transition Chart")
        switch_chart = alt.Chart(switch_df).mark_bar(color="#FF6B6B").encode(
            x=alt.X("Transition:N", sort="-y"),
            y=alt.Y("Count:Q"),
            tooltip=["Transition", "Count"]
        ).properties(height=200)
        st.altair_chart(switch_chart, use_container_width=True)
    
    weekly_switches = weekly_data.groupby("date")["context_switch_count"].sum().reset_index()
    
    st.markdown("### Weekly Context Switch Trend")
    switch_trend = alt.Chart(weekly_switches).mark_line(point=True, color="#FF6B6B").encode(
        x=alt.X("date", title="Date", axis=alt.Axis(format="%d/%m/%Y")),
        y=alt.Y("context_switch_count:Q", title="Switches"),
        tooltip=["date", "context_switch_count"]
    ).properties(height=200)
    st.altair_chart(switch_trend, use_container_width=True)
    
    st.divider()
    
    # ==================== BEHAVIORAL PATTERN (CLUSTER) ====================
    st.markdown("## 🧩 Your Behavioral Pattern")
    
    user_cluster = user_summary[user_summary["user_id"] == user_id]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"### {pattern_label}")
        st.markdown("""
        Based on your app usage patterns across education, social media, and entertainment, 
        we've identified your digital behavior style.
        """)
        
        if "Learner" in pattern_label:
            st.success("You prioritize education and productivity! 📚")
        elif "Butterfly" in pattern_label:
            st.info("You enjoy staying connected with others! 🦋")
        else:
            st.warning("You love your entertainment time! 🎮")
    
    with col2:
        cluster_viz = user_summary[["user_id", "education_minutes", "social_minutes", "entertainment_minutes", "behavior_pattern"]].copy()
        cluster_viz["is_you"] = cluster_viz["user_id"] == user_id
        
        fig = px.scatter(
            cluster_viz,
            x="education_minutes",
            y="entertainment_minutes",
            size="social_minutes",
            color="behavior_pattern",
            hover_data=["user_id"],
            title="Behavioral Pattern Groups (Size = Social Usage)"
        )
        
        you_data = cluster_viz[cluster_viz["is_you"]]
        if not you_data.empty:
            fig.add_trace(go.Scatter(
                x=you_data["education_minutes"],
                y=you_data["entertainment_minutes"],
                mode="markers",
                marker=dict(size=20, symbol="star", color="gold", line=dict(width=2, color="black")),
                name="⭐ You"
            ))
        
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # ==================== NUDGES ====================
    st.markdown("## 💡 Personalized Insights")
    
    nudges = generate_nudges(daily_data, percentile, goal_score, total_context_switches, goal_mode)
    
    for nudge in nudges:
        st.markdown(f"- {nudge}")
    
    st.divider()
    
    # ==================== CHATBOT ====================
    st.markdown("## 💬 Chat with MITRAVA")
    st.markdown("*Your supportive digital wellbeing companion*")
    
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.messages:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"👋 Hey {user_info['name']}! I'm MITRAVA, your digital wellbeing buddy. How are you feeling today? I'm here to help with focus, stress, or just to chat! 💙"
            })
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        response = get_chatbot_response(prompt, percentile, total_context_switches, goal_score)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        with st.chat_message("assistant"):
            st.markdown(response)

# ============================================================
# SECTION 14: MAIN APP ROUTING
# ============================================================
def main():
    """Main application entry point"""
    if not st.session_state.authenticated:
        show_login_page()
    elif st.session_state.is_admin:
        show_admin_panel()
    else:
        show_user_dashboard()

if __name__ == "__main__":
    main()
