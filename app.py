import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
import json
from datetime import datetime
import random

# Page configuration
st.set_page_config(
    page_title="Tamil Dialect Preserver & Storyteller",
    page_icon="📖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B35;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #FFA500, #FF6B35);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .dialect-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: #f0f2f6;
        margin: 1rem 0;
        border-left: 5px solid #FF6B35;
    }
    .story-box {
        padding: 2rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'stories' not in st.session_state:
    st.session_state.stories = []
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {
        'region': 'All',
        'interests': []
    }

# Tamil dialects data
DIALECTS = {
    'Kongu Tamil': {
        'region': 'Coimbatore, Erode',
        'features': 'Unique phonology, distinct vocabulary',
        'sample': 'எங்கட ஊர்ல பழைய கதைகள் நிறைய இருக்கு'
    },
    'Madurai Tamil': {
        'region': 'Madurai, Sivaganga',
        'features': 'Rich in folk expressions, musical intonation',
        'sample': 'நம்ம ஊர்ல பாட்டி சொன்ன கதை கேட்டியா?'
    },
    'Tirunelveli Tamil': {
        'region': 'Tirunelveli, Thoothukudi',
        'features': 'Coastal influences, unique idioms',
        'sample': 'நம்ம கடலோர கதைகள் ரொம்ப சுவாரஸ்யமானவை'
    },
    'Chennai Tamil': {
        'region': 'Chennai Metro',
        'features': 'Urban blend, colloquial expressions',
        'sample': 'நம்ம சென்னை கதைகள் வேற லெவல்'
    }
}

# Sample stories database
SAMPLE_STORIES = [
    {
        'title': 'முயற்சி திருவினையாக்கும்',
        'dialect': 'Kongu Tamil',
        'category': 'Moral Tales',
        'content': '''ஒரு காலத்துல ஒரு சிறு கிராமத்துல ஒரு ஏழை விவசாயி வாழ்ந்தான். அவன் வயல்ல நல்ல பயிர் விளைய ரொம்ப முயற்சி பண்ணினான். 

எல்லாரும் "இந்த மண்ணுல எதுவும் விளையாது"னு சொன்னாங்க. ஆனா அவன் கேக்கலை. நாள் முழுக்க கடுமையா உழைச்சான்.

மூணு வருஷம் கழிச்சு, அவன் வயல் கிராமத்துலேயே சிறந்த விளைச்சலை கொடுத்துச்சு. எல்லாரும் வியந்து போனாங்க.

கதையோட பாடம்: "முயற்சி உடையார் இகழ்ச்சி அடையார்" - விடாமுயற்சி எப்பவும் வெற்றி தரும்.''',
        'moral': 'Perseverance leads to success',
        'proverbs': ['முயற்சி திருவினையாக்கும்', 'முயற்சி உடையார் இகழ்ச்சி அடையார்'],
        'cultural_notes': 'Reflects agricultural community values and work ethic'
    },
    {
        'title': 'புத்திசாலி நரி',
        'dialect': 'Madurai Tamil',
        'category': 'Animal Fables',
        'content': '''ஒரு காட்டுல ஒரு புத்திசாலி நரி இருந்துச்சு. ஒரு நாள் அது ரொம்ப பசியா இருந்துச்சு. திராட்சை கொடியில் நல்ல பழுத்த திராட்சை பழங்கள் தொங்கிக்கிட்டு இருந்தது.

நரி குதிச்சுது, ஆனா எட்டலை. திரும்பவும் முயற்சி பண்ணுச்சு. பல தடவை முயற்சி செஞ்சும் எட்டலை.

கடைசியில நரி "அந்த திராட்சை பழம் புளிச்சிருக்கும், எனக்கு வேண்டாம்"னு சொல்லிட்டு போயிடுச்சு.

பாடம்: நமக்கு கிடைக்காத சாக்குல சாக்கு சொல்றதுக்கு பதிலா மெனக்கெட்டு முயற்சி செய்யனும்.''',
        'moral': 'Do not make excuses for failures',
        'proverbs': ['கிடைக்காதது புளிக்கும்'],
        'cultural_notes': 'Classic fable adapted to Tamil cultural context'
    },
    {
        'title': 'கடலோர மீனவன் கதை',
        'dialect': 'Tirunelveli Tamil',
        'category': 'Coastal Tales',
        'content': '''நம்ம கடலோர கிராமத்துல ஒரு மீனவன் இருந்தான். அவன் தினமும் கடல்ல மீன் பிடிக்க போவான்.

ஒரு நாள் ரொம்ப பெரிய புயல் வந்துச்சு. எல்லாரும் "இன்னிக்கு கடல்ல போகாதே"னு சொன்னாங்க. ஆனா அவன் "என் குடும்பத்துக்கு சாப்பாடு வேணும்"னு போனான்.

கடல்ல ஒரு பெரிய மீனை பிடிச்சான். ஆனா அதே நேரத்துல அவன் படகு கவிழ போகுது. அப்ப ஒரு டால்பின் வந்து அவனை காப்பாத்துச்சு.

அன்னிக்கு அவன் கத்துக்கிட்டான் - தைரியம் நல்லது, ஆனா இயற்கையை மதிக்கனும்.''',
        'moral': 'Respect nature while being brave',
        'proverbs': ['தைரியம் நல்லது, ஆபத்தை அறிவது அவசியம்'],
        'cultural_notes': 'Reflects fishing community lifestyle and ocean wisdom'
    }
]

def load_model():
    """Load a lightweight model for demonstration"""
    try:
        # Using a smaller model for demonstration
        # In production, use fine-tuned Tamil models
        with st.spinner('Loading AI model...'):
            model_name = "gpt2"  # Placeholder - replace with Tamil-specific model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None

def translate_to_english(tamil_text):
    """Simulate translation - in production use MarianMT or similar"""
    translations = {
        'முயற்சி திருவினையாக்கும்': 'Effort brings success',
        'புத்திசாலி நரி': 'The Clever Fox',
        'கடலோர மீனவன் கதை': 'The Coastal Fisherman Tale'
    }
    return translations.get(tamil_text, "Translation: " + tamil_text)

def recommend_stories(preferences):
    """Recommend stories based on user preferences"""
    recommended = []
    for story in SAMPLE_STORIES:
        if preferences['region'] == 'All' or story['dialect'] == preferences['region']:
            if not preferences['interests'] or story['category'] in preferences['interests']:
                recommended.append(story)
    return recommended if recommended else SAMPLE_STORIES

def main():
    # Header
    st.markdown('<h1 class="main-header">🎭 Tamil Dialect Preserver & Storyteller 📖</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Cultural Heritage Platform")
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/Tamil_Language.svg/120px-Tamil_Language.svg.png", width=100)
        st.title("Navigation")
        page = st.radio("", ["🏠 Home", "📚 Story Library", "🎙️ Record Story", "🔍 Dialect Explorer", "📊 Analytics"])
        
        st.markdown("---")
        st.subheader("User Preferences")
        region = st.selectbox("Select Region", ['All'] + list(DIALECTS.keys()))
        interests = st.multiselect("Interests", ['Moral Tales', 'Animal Fables', 'Coastal Tales', 'Historical Stories'])
        
        st.session_state.user_preferences = {
            'region': region,
            'interests': interests
        }
    
    # Main content based on page selection
    if page == "🏠 Home":
        show_home()
    elif page == "📚 Story Library":
        show_library()
    elif page == "🎙️ Record Story":
        show_recorder()
    elif page == "🔍 Dialect Explorer":
        show_dialect_explorer()
    elif page == "📊 Analytics":
        show_analytics()

def show_home():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🌟 Welcome to Tamil Dialect Preservation Platform")
        st.markdown("""
        This AI-powered platform preserves endangered Tamil dialects through:
        - 🎙️ **Community Recording**: Elders share folk stories in their native dialect
        - 🤖 **AI Processing**: ASR and NLP for transcription and translation
        - 📖 **Storytelling**: Curated stories with cultural annotations
        - 🔍 **Research**: Archive for linguistic and cultural study
        """)
        
        st.markdown("### 🎯 Featured Story of the Day")
        featured = random.choice(SAMPLE_STORIES)
        with st.container():
            st.markdown(f"### {featured['title']}")
            st.markdown(f"**Dialect**: {featured['dialect']} | **Category**: {featured['category']}")
            with st.expander("Read Story"):
                st.markdown(featured['content'])
                st.markdown(f"**Moral**: {featured['moral']}")
    
    with col2:
        st.markdown("### 📊 Platform Stats")
        st.metric("Stories Archived", "847")
        st.metric("Dialects Covered", "12")
        st.metric("Community Contributors", "234")
        st.metric("Active Users", "1,523")
        
        st.markdown("### 🗺️ Dialect Coverage")
        for dialect in list(DIALECTS.keys())[:3]:
            st.progress(random.randint(60, 95)/100, text=dialect)

def show_library():
    st.markdown("## 📚 Story Library")
    
    # Get recommended stories
    recommended = recommend_stories(st.session_state.user_preferences)
    
    st.markdown(f"### Showing {len(recommended)} stories based on your preferences")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        search = st.text_input("🔍 Search stories", "")
    with col2:
        category_filter = st.selectbox("Category", ['All'] + ['Moral Tales', 'Animal Fables', 'Coastal Tales'])
    with col3:
        sort_by = st.selectbox("Sort by", ['Relevance', 'Title', 'Dialect'])
    
    # Display stories
    for story in recommended:
        if search.lower() in story['title'].lower() or search == "":
            if category_filter == 'All' or category_filter == story['category']:
                with st.container():
                    st.markdown('<div class="story-box">', unsafe_allow_html=True)
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"### 📖 {story['title']}")
                        st.markdown(f"**Dialect**: {story['dialect']} | **Category**: {story['category']}")
                    
                    with col2:
                        if st.button("Read", key=story['title']):
                            st.session_state.selected_story = story
                    
                    with st.expander("View Story Details"):
                        st.markdown("#### Tamil Text")
                        st.markdown(story['content'])
                        
                        st.markdown("#### English Translation")
                        st.info(f"Translation: {story['moral']}")
                        
                        st.markdown("#### Proverbs Used")
                        for proverb in story['proverbs']:
                            st.markdown(f"- {proverb} ({translate_to_english(proverb)})")
                        
                        st.markdown("#### Cultural Notes")
                        st.markdown(story['cultural_notes'])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.button("🔊 Listen", key=f"listen_{story['title']}")
                        with col2:
                            st.button("💾 Save", key=f"save_{story['title']}")
                        with col3:
                            st.button("📤 Share", key=f"share_{story['title']}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)

def show_recorder():
    st.markdown("## 🎙️ Record Your Story")
    st.markdown("Help preserve your dialect by recording folk stories, proverbs, or idioms!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Recording Interface")
        
        story_title = st.text_input("Story Title", "")
        dialect = st.selectbox("Select Your Dialect", list(DIALECTS.keys()))
        category = st.selectbox("Category", ['Folk Tale', 'Proverb', 'Idiom', 'Historical Story', 'Other'])
        
        st.markdown("#### Record Audio")
        st.info("🎙️ Click the button below to start recording (Demo Mode)")
        
        if st.button("🔴 Start Recording"):
            st.success("Recording started! (Demo)")
            st.audio("https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3")
        
        st.markdown("#### Or Upload Audio File")
        audio_file = st.file_uploader("Upload Audio (.mp3, .wav)", type=['mp3', 'wav'])
        
        if audio_file:
            st.audio(audio_file)
            
            if st.button("🤖 Process with AI"):
                with st.spinner("Processing audio with ASR..."):
                    st.success("✅ Transcription Complete!")
                    st.markdown("### Transcribed Text")
                    st.text_area("Tamil Text", "ஒரு காலத்துல ஒரு கிராமத்துல...", height=200)
                    
                    st.markdown("### AI Analysis")
                    st.markdown("**Detected Dialect**: " + dialect)
                    st.markdown("**Phonological Features**: Unique vowel lengthening patterns detected")
                    st.markdown("**Idiomatic Expressions**: 2 regional idioms identified")
                    
                    if st.button("💾 Save to Archive"):
                        st.success("Story saved successfully! 🎉")
    
    with col2:
        st.markdown("### Recording Guidelines")
        st.markdown("""
        ✅ **Best Practices**:
        - Find a quiet location
        - Speak clearly and naturally
        - Include context about the story
        - Mention any special dialect words
        
        📝 **What to Record**:
        - Traditional folk stories
        - Proverbs and their meanings
        - Regional idioms
        - Historical narratives
        - Cultural practices
        """)
        
        st.markdown("### 🏆 Top Contributors")
        contributors = [
            {"name": "Lakshmi Patti", "stories": 23},
            {"name": "Raman Thatha", "stories": 18},
            {"name": "Meena Akka", "stories": 15}
        ]
        for c in contributors:
            st.markdown(f"**{c['name']}**: {c['stories']} stories")

def show_dialect_explorer():
    st.markdown("## 🔍 Dialect Explorer")
    st.markdown("Explore the rich diversity of Tamil dialects across regions")
    
    # Dialect selection
    selected_dialect = st.selectbox("Choose a Dialect", list(DIALECTS.keys()))
    
    dialect_info = DIALECTS[selected_dialect]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f'<div class="dialect-card">', unsafe_allow_html=True)
        st.markdown(f"## {selected_dialect}")
        st.markdown(f"**Region**: {dialect_info['region']}")
        st.markdown(f"**Features**: {dialect_info['features']}")
        st.markdown(f"### Sample Text")
        st.code(dialect_info['sample'], language="text")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### Phonological Features")
        st.markdown("""
        - **Vowel System**: Unique long/short vowel patterns
        - **Consonant Variations**: Regional pronunciation differences
        - **Intonation**: Distinct melodic patterns
        - **Stress Patterns**: Word-level emphasis variations
        """)
        
        st.markdown("### Common Expressions")
        expressions = [
            {"tamil": "எப்படி இருக்கீங்க?", "meaning": "How are you?", "standard": "எப்படி இருக்கிறீர்கள்?"},
            {"tamil": "எங்கட ஊர்", "meaning": "Our village", "standard": "எங்கள் ஊர்"},
            {"tamil": "நல்லா இருக்கு", "meaning": "It's good", "standard": "நன்றாக இருக்கிறது"}
        ]
        
        for exp in expressions:
            with st.expander(f"{exp['tamil']}"):
                st.markdown(f"**Meaning**: {exp['meaning']}")
                st.markdown(f"**Standard Tamil**: {exp['standard']}")
    
    with col2:
        st.markdown("### 🗺️ Geographic Distribution")
        st.info(f"Primarily spoken in: {dialect_info['region']}")
        
        st.markdown("### 📊 Vitality Status")
        vitality = random.randint(40, 80)
        st.progress(vitality/100)
        st.markdown(f"Vitality Score: {vitality}%")
        
        st.markdown("### 📚 Resources")
        st.markdown(f"- Stories in {selected_dialect}: {random.randint(20, 80)}")
        st.markdown(f"- Audio Recordings: {random.randint(50, 150)}")
        st.markdown(f"- Annotated Texts: {random.randint(10, 40)}")
        
        if st.button("📥 Download Dialect Pack"):
            st.success("Dialect resource pack downloaded!")

def show_analytics():
    st.markdown("## 📊 Platform Analytics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Stories", "847", "+23 this week")
    with col2:
        st.metric("Active Dialects", "12", "+1 new")
    with col3:
        st.metric("Contributors", "234", "+15")
    with col4:
        st.metric("Total Hours", "1,234", "+87 hrs")
    
    st.markdown("### 📈 Story Uploads Over Time")
    import pandas as pd
    import numpy as np
    
    # Generate sample data
    dates = pd.date_range(start='2024-01-01', end='2024-10-28', freq='W')
    uploads = np.random.randint(5, 25, size=len(dates))
    df = pd.DataFrame({'Date': dates, 'Stories': uploads})
    
    st.line_chart(df.set_index('Date'))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🗣️ Dialect Distribution")
        dialect_data = {dialect: random.randint(30, 120) for dialect in DIALECTS.keys()}
        st.bar_chart(dialect_data)
    
    with col2:
        st.markdown("### 📚 Category Breakdown")
        category_data = {
            'Moral Tales': 234,
            'Animal Fables': 189,
            'Coastal Tales': 156,
            'Historical': 142,
            'Others': 126
        }
        st.bar_chart(category_data)
    
    st.markdown("### 🌍 Geographic Reach")
    st.map(pd.DataFrame({
        'lat': [11.0168, 9.9252, 10.7905, 13.0827],
        'lon': [76.9558, 78.1198, 78.7047, 80.2707],
        'size': [100, 80, 70, 90]
    }))
    
    st.markdown("### 🏆 Top Contributing Regions")
    regions = [
        {"region": "Coimbatore", "stories": 156, "contributors": 45},
        {"region": "Madurai", "stories": 142, "contributors": 38},
        {"region": "Tirunelveli", "stories": 128, "contributors": 32},
        {"region": "Chennai", "stories": 98, "contributors": 28}
    ]
    
    for r in regions:
        with st.expander(f"📍 {r['region']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Stories", r['stories'])
            with col2:
                st.metric("Contributors", r['contributors'])

if __name__ == "__main__":
    main()