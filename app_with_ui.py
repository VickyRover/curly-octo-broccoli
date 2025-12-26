import streamlit as st
import streamlit.components.v1 as components
import os

# Page configuration
st.set_page_config(
    page_title="Tamil Dialect AI - Interactive",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Navigation
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/Tamil_Language.svg/120px-Tamil_Language.svg.png", width=100)
    st.title("🎭 Tamil Dialect AI")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Interactive Home", "📚 Story Library", "🎙️ Record Story", 
         "🔍 Dialect Explorer", "📊 Analytics", "🤖 AI Demo"]
    )
    
    st.markdown("---")
    st.markdown("### Quick Stats")
    st.metric("Stories", "847", "+23")
    st.metric("Dialects", "12", "+1")
    st.metric("Users", "1,523", "+87")

# Main Content based on page selection
if page == "🏠 Interactive Home":
    # Embed the interactive HTML page
    html_file_path = "interactive_ui.html"
    
    # Check if HTML file exists
    if os.path.exists(html_file_path):
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Display HTML in full width
        components.html(html_content, height=3000, scrolling=True)
    else:
        st.error("⚠️ Interactive UI file not found! Please create 'interactive_ui.html'")
        st.info("👉 Copy the HTML code from the artifact and save it as 'interactive_ui.html' in your project folder")
        
        # Provide download button
        if st.button("📥 Show HTML Code"):
            st.code("""
            <!-- Copy the complete HTML code from the interactive UI artifact
                 and save it as 'interactive_ui.html' in your project folder -->
            """, language="html")

elif page == "📚 Story Library":
    st.title("📚 Story Library")
    st.markdown("Browse our collection of Tamil dialect stories")
    
    # Your existing story library code here
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🏞️ Kongu Tamil")
        st.info("Stories: 156")
        if st.button("View Stories", key="kongu"):
            st.success("Loading Kongu Tamil stories...")
    
    with col2:
        st.markdown("### 🎭 Madurai Tamil")
        st.info("Stories: 142")
        if st.button("View Stories", key="madurai"):
            st.success("Loading Madurai Tamil stories...")
    
    with col3:
        st.markdown("### 🌊 Tirunelveli Tamil")
        st.info("Stories: 128")
        if st.button("View Stories", key="tirunelveli"):
            st.success("Loading Tirunelveli Tamil stories...")

elif page == "🎙️ Record Story":
    st.title("🎙️ Record Your Story")
    st.markdown("Help preserve your dialect by recording folk stories!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        story_title = st.text_input("Story Title", placeholder="Enter story title in Tamil")
        dialect = st.selectbox("Select Dialect", ["Kongu Tamil", "Madurai Tamil", "Tirunelveli Tamil", "Chennai Tamil"])
        category = st.selectbox("Category", ["Folk Tale", "Proverb", "Moral Story", "Historical"])
        
        st.markdown("### Record Audio")
        audio_file = st.file_uploader("Upload Audio File", type=['mp3', 'wav', 'ogg'])
        
        if audio_file:
            st.audio(audio_file)
            
            if st.button("🤖 Process with AI"):
                with st.spinner("Processing..."):
                    import time
                    time.sleep(2)
                    st.success("✅ Story recorded successfully!")
                    st.balloons()
    
    with col2:
        st.info("""
        ### 📝 Guidelines
        - Speak clearly
        - Quiet environment
        - Include context
        - Use native dialect
        """)

elif page == "🔍 Dialect Explorer":
    st.title("🔍 Dialect Explorer")
    
    dialects = {
        "Kongu Tamil": {"region": "Coimbatore, Erode", "icon": "🏞️"},
        "Madurai Tamil": {"region": "Madurai, Sivaganga", "icon": "🎭"},
        "Tirunelveli Tamil": {"region": "Tirunelveli Coast", "icon": "🌊"},
        "Chennai Tamil": {"region": "Chennai Metro", "icon": "🏙️"}
    }
    
    selected = st.selectbox("Choose a Dialect", list(dialects.keys()))
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"## {dialects[selected]['icon']} {selected}")
        st.markdown(f"**Region:** {dialects[selected]['region']}")
        st.markdown("### Sample Expression")
        st.code("எங்கட ஊர்ல நல்ல கதைகள் நிறைய இருக்கு", language="text")
        st.markdown("**Meaning:** There are many good stories in our village")
    
    with col2:
        st.metric("Vitality Score", "75%")
        st.metric("Stories Archived", "156")
        st.metric("Active Contributors", "45")

elif page == "📊 Analytics":
    st.title("📊 Platform Analytics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Stories", "847", "+23")
    col2.metric("Dialects", "12", "+1")
    col3.metric("Contributors", "234", "+15")
    col4.metric("Hours Recorded", "1,234", "+87")
    
    st.markdown("### 📈 Story Uploads Over Time")
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range(start='2024-01-01', end='2024-10-29', freq='W')
    data = pd.DataFrame({
        'Date': dates,
        'Stories': np.random.randint(5, 25, size=len(dates))
    })
    
    st.line_chart(data.set_index('Date'))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🗣️ Dialect Distribution")
        dialect_data = {
            'Kongu Tamil': 156,
            'Madurai Tamil': 142,
            'Tirunelveli Tamil': 128,
            'Chennai Tamil': 98
        }
        st.bar_chart(dialect_data)
    
    with col2:
        st.markdown("### 📚 Category Breakdown")
        category_data = {
            'Moral Tales': 234,
            'Animal Fables': 189,
            'Coastal Tales': 156,
            'Historical': 142
        }
        st.bar_chart(category_data)

elif page == "🤖 AI Demo":
    st.title("🤖 AI Story Generator Demo")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Generate New Story")
        dialect = st.selectbox("Select Dialect", ["Kongu Tamil", "Madurai Tamil", "Tirunelveli Tamil", "Chennai Tamil"], key="ai_dialect")
        theme = st.selectbox("Choose Theme", ["Moral Tale", "Animal Fable", "Coastal Tale"])
        length = st.slider("Story Length", 100, 500, 200)
        
        if st.button("🎲 Generate Story", type="primary"):
            with st.spinner("AI is generating your story..."):
                import time
                time.sleep(2)
                
                st.success("Story generated!")
                
                st.markdown("### முயற்சியின் வெற்றி")
                st.markdown("""
                ஒரு காலத்தில் ஒரு கிராமத்தில் ஒரு விவசாயி வாழ்ந்தார். 
                அவர் கடுமையாக உழைத்தார். மூன்று ஆண்டுகள் கழித்து அவரது 
                வயல் சிறந்த விளைச்சலை அளித்தது.
                
                **பாடம்:** விடாமுயற்சி வெற்றியைத் தரும்
                """)
                
                col_a, col_b, col_c = st.columns(3)
                col_a.button("🔊 Listen")
                col_b.button("💾 Save")
                col_c.button("📤 Share")
    
    with col2:
        st.markdown("### 🎯 AI Features")
        st.info("✅ Dialect-specific generation")
        st.info("✅ Cultural authenticity")
        st.info("✅ Proverb integration")
        st.info("✅ Moral lessons")
        
        st.markdown("### 📊 Model Info")
        st.code("""
        Model: LLaMA with LoRA
        Parameters: 7B
        Training: Tamil dialects
        Accuracy: 89%
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎭 Tamil Dialect AI Project | VISTAS, Chennai</p>
    <p>Preserving heritage through technology</p>
</div>
""", unsafe_allow_html=True)