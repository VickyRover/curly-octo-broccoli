# 🎭 Tamil Dialect AI Project - Complete Setup Guide

## Project Overview

An AI-powered platform for preserving Tamil dialects through community storytelling, featuring:
- 🎙️ Audio recording & ASR processing
- 🤖 NLP-based dialect translation
- 📖 AI-powered story generation
- 🔍 Dialect exploration & analytics
- 📊 Community engagement dashboard

---

## 📁 Project Structure

```
tamil-dialect-ai/
│
├── app.py                    # Main Streamlit application
├── generate_story.py         # AI story generator
├── train_tamil_2.py          # Model training with LoRA
├── tamil_stories1.txt        # Sample Tamil stories dataset
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🚀 Quick Start

### Step 1: Install Dependencies

Create a `requirements.txt` file:

```txt
streamlit==1.28.0
torch==2.0.1
transformers==4.35.0
peft==0.6.0
datasets==2.14.0
pandas==2.1.0
numpy==1.24.3
```

Install:
```bash
pip install -r requirements.txt
```

### Step 2: Run the Main Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Step 3: Generate Stories (Optional)

```bash
python generate_story.py
```

This creates:
- `generated_stories.json` - JSON format stories
- `tamil_stories1.txt` - Text format stories

### Step 4: Train Model (Advanced)

```bash
python train_tamil_2.py
```

This will:
- Fine-tune a model with LoRA
- Save to `./tamil_dialect_model/`
- Generate test outputs

---

## 💻 Detailed Installation Guide

### For Windows:

```bash
# 1. Install Python 3.8+ from python.org

# 2. Create virtual environment
python -m venv tamil_env

# 3. Activate environment
tamil_env\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run application
streamlit run app.py
```

### For Mac/Linux:

```bash
# 1. Create virtual environment
python3 -m venv tamil_env

# 2. Activate environment
source tamil_env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run application
streamlit run app.py
```

### For Google Colab (Cloud):

```python
# In a Colab notebook:
!pip install streamlit transformers peft datasets

# Upload your files
from google.colab import files
files.upload()  # Upload app.py, generate_story.py, etc.

# Run with ngrok tunnel
!streamlit run app.py & npx localtunnel --port 8501
```

---

## 📖 Feature Guide

### 1. Home Page
- Platform statistics
- Featured story of the day
- Dialect coverage metrics

### 2. Story Library
- Browse stories by dialect
- Filter by category
- Search functionality
- Cultural annotations
- Audio playback (demo)

### 3. Record Story
- Audio recording interface
- File upload support
- AI-powered transcription
- Dialect detection
- Contribution tracking

### 4. Dialect Explorer
- 12 Tamil dialects covered
- Phonological features
- Sample expressions
- Geographic distribution
- Vitality scores

### 5. Analytics Dashboard
- Upload trends
- Dialect distribution
- Geographic reach map
- Top contributors

---

## 🤖 AI Components

### Speech Recognition (ASR)
```python
# Current: Demo mode
# Production: Use Whisper, Wav2Vec2, or custom Tamil ASR
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="openai/whisper-base")
```

### Language Model (Story Generation)
```python
# Current: GPT-2 (demo)
# Production: Use ai4bharat/IndicBART or fine-tuned LLaMA
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("ai4bharat/IndicBART")
```

### Translation
```python
# Production: Use MarianMT for Tamil-English translation
from transformers import MarianMTModel, MarianTokenizer
model_name = "Helsinki-NLP/opus-mt-ta-en"
```

### LoRA Fine-tuning
```python
# Efficient fine-tuning with minimal parameters
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05)
```

---

## 🔧 Customization Guide

### Adding New Dialects

In `app.py`, update the `DIALECTS` dictionary:

```python
DIALECTS = {
    'Your New Dialect': {
        'region': 'Geographic Region',
        'features': 'Unique characteristics',
        'sample': 'Sample text in dialect'
    }
}
```

### Adding New Stories

In `generate_story.py`, add to templates:

```python
"your_theme": {
    "title": "Story Title",
    "content": """Story content with {variables}""",
    "moral": "Moral of the story",
    "proverbs": ["Proverb 1", "Proverb 2"]
}
```

### Custom Model Integration

Replace placeholder models in `train_tamil_2.py`:

```python
# Use Tamil-specific models
base_model = "ai4bharat/IndicBART"  # For Indian languages
# Or
base_model = "meta-llama/Llama-2-7b-hf"  # For larger scale
```

---

## 📊 Sample Data

The project includes sample stories in multiple Tamil dialects:

1. **Kongu Tamil** - Agricultural themes
2. **Madurai Tamil** - Folk expressions
3. **Tirunelveli Tamil** - Coastal narratives
4. **Chennai Tamil** - Urban stories

Each story includes:
- Original Tamil text
- English translation
- Cultural notes
- Relevant proverbs
- Moral lessons

---

## 🎯 Production Deployment

### For Research/Demo:
```bash
streamlit run app.py --server.port 8501 --server.address localhost
```

### For Production:
```bash
# Use Streamlit Cloud, Heroku, or AWS
# Add authentication
# Configure database (PostgreSQL/MongoDB)
# Set up CDN for audio files
# Implement proper ASR pipeline
```

### Environment Variables:
```bash
# Create .env file
HUGGINGFACE_TOKEN=your_token
AWS_ACCESS_KEY=your_key
DATABASE_URL=your_db_url
```

---

## 🐛 Troubleshooting

### Common Issues:

**1. ModuleNotFoundError**
```bash
# Solution: Install missing package
pip install <package_name>
```

**2. CUDA out of memory**
```python
# Solution: Use CPU or reduce batch size
device = torch.device("cpu")
batch_size = 1
```

**3. Streamlit won't start**
```bash
# Solution: Check port availability
streamlit run app.py --server.port 8502
```

**4. Model download fails**
```bash
# Solution: Set HuggingFace cache
export TRANSFORMERS_CACHE=/path/to/cache
```

---

## 📚 Academic References

1. **Speech Recognition**
   - Nanmalar et al., "Literary and Colloquial Tamil Dialect Identification," arXiv:2408.13739, 2024

2. **Dialect Processing**
   - Saranya et al., "Real-Time Tamil Dialect Speech Recognition using LoRA," Signal Processing, 2025

3. **Cultural AI**
   - Ghosh, "Role of AI in Transmitting Indian Culture," IAJESM, 2019

---

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Submit a pull request

Focus areas:
- Adding more dialects
- Improving ASR accuracy
- Creating more stories
- Enhancing UI/UX
- Optimizing model performance

---

## 📄 License
This project is for academic and research purposes. Please cite appropriately if used in publications.

---

## 👥 Team

**VIGNESHWARAN M, YOKESH SK**  
UG Scholars, Department of B.Tech CSE (Data Science) 
Vels Institute of Science Technology and Advanced Studies (VISTAS), Chennai

**Guide: MRS.V BHARATHI**  
Associate Professor, Department of  Computer Science Engineering
VISTAS, Chennai

---

## 🎓 Project Goals

1. **Preservation**: Archive endangered Tamil dialects
2. **Education**: Teach younger generations about linguistic diversity
3. **Research**: Provide dataset for Tamil NLP research
4. **Community**: Engage elders and communities in preservation
5. **Innovation**: Demonstrate AI for cultural heritage

---