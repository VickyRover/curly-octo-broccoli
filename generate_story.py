import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer
import json
import random
from datetime import datetime

class TamilStoryGenerator:
    def __init__(self, model_name="gpt2"):
        """
        Initialize the story generator
        In production, replace with Tamil-specific models like:
        - "ai4bharat/IndicBART"
        - Custom fine-tuned LLaMA with LoRA
        """
        print("Initializing Tamil Story Generator...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.tokenizer = None
            self.model = None
    
    def generate_story(self, 
                      dialect="Kongu Tamil",
                      theme="moral tale",
                      length="medium",
                      temperature=0.8):
        """
        Generate a Tamil story based on parameters
        """
        if self.model is None:
            return self._generate_template_story(dialect, theme)
        
        # Story prompts for different dialects
        prompts = {
            "Kongu Tamil": "ஒரு காலத்துல எங்கட ஊர்ல ",
            "Madurai Tamil": "நம்ம ஊர்ல பழைய காலத்துல ",
            "Tirunelveli Tamil": "கடலோர கிராமத்துல ஒரு நாள் ",
            "Chennai Tamil": "சென்னையில் ஒரு பையன் "
        }
        
        prompt = prompts.get(dialect, "ஒரு காலத்தில் ")
        
        # Generate using model (simplified for demo)
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=200,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    num_return_sequences=1
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Create story structure
            story = {
                "title": self._generate_title(theme, dialect),
                "dialect": dialect,
                "theme": theme,
                "content": generated_text,
                "moral": self._extract_moral(theme),
                "proverbs": self._get_relevant_proverbs(theme),
                "cultural_notes": self._generate_cultural_notes(dialect, theme),
                "generated_at": datetime.now().isoformat()
            }
            
            return story
            
        except Exception as e:
            print(f"Generation error: {e}")
            return self._generate_template_story(dialect, theme)
    
    def _generate_template_story(self, dialect, theme):
        """Generate a template-based story when model is unavailable"""
        
        templates = {
            "moral tale": {
                "title": "முயற்சியின் வெற்றி",
                "content": """ஒரு காலத்தில் {location} ஒரு {character} வாழ்ந்தார். 
                
அவர் மிகவும் {quality} கொண்டவர். ஒரு நாள் அவர் {challenge} எதிர்கொண்டார்.

எல்லோரும் "இது சாத்தியமில்லை" என்று சொன்னார்கள். ஆனால் அவர் விடாமல் முயற்சி செய்தார்.

{duration} கழித்து, அவர் வெற்றி பெற்றார். கிராமத்தார் அனைவரும் வியந்தனர்.

இக்கதையின் பாடம்: {moral}""",
                "moral": "விடாமுயற்சி வெற்றியைத் தரும்",
                "proverbs": ["முயற்சி திருவினையாக்கும்", "முயற்சி உடையார் இகழ்ச்சி அடையார்"]
            },
            "animal fable": {
                "title": "{animal} கதை",
                "content": """காட்டில் ஒரு {animal} வாழ்ந்தது. அது மிகவும் {trait} கொண்டது.

ஒரு நாள் அது {situation} சந்தித்தது. {animal} {action} செய்தது.

இதனால் {consequence}. மற்ற விலங்குகள் {reaction}.

பாடம்: {moral}""",
                "moral": "புத்திசாலித்தனம் பலத்தை விட சிறந்தது",
                "proverbs": ["புத்தி பலத்தை வெல்லும்", "சிறுத்தையும் புத்தியால் வெல்லலாம்"]
            },
            "coastal tale": {
                "title": "கடலோர {character}",
                "content": """கடலோர கிராமத்தில் ஒரு {character} வாழ்ந்தார். கடல் அவரது வாழ்க்கை.

ஒரு நாள் பெரிய புயல் வந்தது. {character} {action}.

கடல் {event}. ஆனால் {character} {response}.

இறுதியில் {outcome}. கிராமத்தார் {conclusion}.

பாடம்: {moral}""",
                "moral": "இயற்கையை மதிக்க வேண்டும்",
                "proverbs": ["கடல் நமது தாய்", "இயற்கை நமது குரு"]
            }
        }
        
        template = templates.get(theme, templates["moral tale"])
        
        # Fill template with variables
        variables = {
            "location": random.choice(["ஒரு கிராமத்தில்", "ஒரு நகரத்தில்", "மலையடிவாரத்தில்"]),
            "character": random.choice(["விவசாயி", "வணிகர்", "மீனவர்", "சிறுவன்", "முதியவர்"]),
            "quality": random.choice(["உழைப்பாளி", "புத்திசாலி", "நேர்மையானவர்", "தைரியமானவர்"]),
            "challenge": random.choice(["ஒரு பெரிய பிரச்சனையை", "கடினமான சவாலை", "ஒரு இடையூறை"]),
            "duration": random.choice(["சில மாதங்கள்", "ஒரு வருடம்", "பல நாட்கள்"]),
            "moral": template["moral"],
            "animal": random.choice(["நரி", "முயல்", "யானை", "குரங்கு"]),
            "trait": random.choice(["புத்திசாலி", "வலிமையான", "விரைவான"]),
            "situation": random.choice(["ஒரு சிக்கலை", "ஆபத்தை", "வாய்ப்பை"]),
            "action": random.choice(["புத்திசாலித்தனமாக செயல்பட்டது", "யோசித்து முடிவெடுத்தது"]),
            "consequence": random.choice(["நல்ல பலன் கிடைத்தது", "வெற்றி அடைந்தது"]),
            "reaction": random.choice(["மகிழ்ந்தன", "பாராட்டின"]),
            "event": random.choice(["கோபமாக இருந்தது", "அமைதியானது"]),
            "response": random.choice(["தைரியமாக செயல்பட்டார்", "புத்திசாலித்தனமாக நடந்தார்"]),
            "outcome": random.choice(["நல்ல முடிவு கிடைத்தது", "அனைவரும் மகிழ்ந்தனர்"]),
            "conclusion": random.choice(["பாராட்டினார்கள்", "கொண்டாடினார்கள்"])
        }
        
        content = template["content"]
        for key, value in variables.items():
            content = content.replace("{" + key + "}", value)
        
        title = template["title"]
        for key, value in variables.items():
            title = title.replace("{" + key + "}", value)
        
        return {
            "title": title,
            "dialect": dialect,
            "theme": theme,
            "content": content,
            "moral": template["moral"],
            "proverbs": template["proverbs"],
            "cultural_notes": self._generate_cultural_notes(dialect, theme),
            "generated_at": datetime.now().isoformat()
        }
    
    def _generate_title(self, theme, dialect):
        """Generate a title based on theme"""
        titles = {
            "moral tale": ["உழைப்பின் பலன்", "நேர்மையின் வெற்றி", "முயற்சியின் பலம்"],
            "animal fable": ["புத்திசாலி விலங்கு", "காட்டு நண்பர்கள்", "விலங்குகள் கதை"],
            "coastal tale": ["கடலோர வாழ்க்கை", "மீனவர் கதை", "கடலின் பாடம்"]
        }
        return random.choice(titles.get(theme, ["தமிழ் கதை"]))
    
    def _extract_moral(self, theme):
        """Extract moral based on theme"""
        morals = {
            "moral tale": "விடாமுயற்சி வெற்றியைத் தரும்",
            "animal fable": "புத்திசாலித்தனம் பலத்தை விட சிறந்தது",
            "coastal tale": "இயற்கையை மதிக்க வேண்டும்"
        }
        return morals.get(theme, "நல்லது செய், நல்லதே நடக்கும்")
    
    def _get_relevant_proverbs(self, theme):
        """Get relevant Tamil proverbs"""
        proverbs = {
            "moral tale": [
                "முயற்சி திருவினையாக்கும்",
                "முயற்சி உடையார் இகழ்ச்சி அடையார்",
                "ஊழியும் உப்பக்கம் காண்பர் உலைவின்றி"
            ],
            "animal fable": [
                "புத்தி பலத்தை வெல்லும்",
                "அறிவுடையார் எல்லாம் உடையார்"
            ],
            "coastal tale": [
                "கடல் நமது தாய்",
                "இயற்கையை மதி"
            ]
        }
        return proverbs.get(theme, ["நல்லது செய், நல்லதே நடக்கும்"])
    
    def _generate_cultural_notes(self, dialect, theme):
        """Generate cultural context notes"""
        notes = {
            "Kongu Tamil": "கோவை மற்றும் சுற்றுவட்டார பகுதிகளில் பேசப்படும் இந்த பேச்சுவழக்கு விவசாய சமூகத்தின் மதிப்புகளை பிரதிபலிக்கிறது.",
            "Madurai Tamil": "மதுரை பகுதியின் வளமான கலாச்சார பாரம்பரியத்தை பிரதிபலிக்கும் இந்த பேச்சுவழக்கு இசை மற்றும் கலையுடன் இணைந்துள்ளது.",
            "Tirunelveli Tamil": "கடலோர மற்றும் விவசாய சமூகங்களின் கலவையான இந்த பேச்சுவழக்கு தனித்துவமான சொற்றொடர்களைக் கொண்டுள்ளது.",
            "Chennai Tamil": "நகர்ப்புற மற்றும் பாரம்பரிய தமிழின் கலவையான இந்த பேச்சுவழக்கு நவீன வாழ்க்கையை பிரதிபலிக்கிறது."
        }
        return notes.get(dialect, "இந்த கதை தமிழ் கலாச்சாரத்தின் பாரம்பரிய மதிப்புகளை பிரதிபலிக்கிறது.")
    
    def batch_generate_stories(self, count=10, dialects=None, themes=None):
        """Generate multiple stories"""
        if dialects is None:
            dialects = ["Kongu Tamil", "Madurai Tamil", "Tirunelveli Tamil", "Chennai Tamil"]
        
        if themes is None:
            themes = ["moral tale", "animal fable", "coastal tale"]
        
        stories = []
        print(f"\nGenerating {count} stories...")
        
        for i in range(count):
            dialect = random.choice(dialects)
            theme = random.choice(themes)
            
            print(f"Generating story {i+1}/{count} - {dialect}, {theme}")
            story = self.generate_story(dialect=dialect, theme=theme)
            stories.append(story)
        
        print(f"\n✓ Generated {len(stories)} stories successfully!")
        return stories
    
    def save_stories(self, stories, filename="generated_stories.json"):
        """Save generated stories to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(stories, f, ensure_ascii=False, indent=2)
        print(f"✓ Stories saved to {filename}")
    
    def export_to_txt(self, stories, filename="tamil_stories.txt"):
        """Export stories to text format"""
        with open(filename, 'w', encoding='utf-8') as f:
            for i, story in enumerate(stories, 1):
                f.write(f"{'='*60}\n")
                f.write(f"கதை {i}: {story['title']}\n")
                f.write(f"{'='*60}\n\n")
                f.write(f"பேச்சுவழக்கு: {story['dialect']}\n")
                f.write(f"வகை: {story['theme']}\n\n")
                f.write(f"கதை:\n{story['content']}\n\n")
                f.write(f"பாடம்: {story['moral']}\n\n")
                f.write(f"பழமொழிகள்:\n")
                for proverb in story['proverbs']:
                    f.write(f"  - {proverb}\n")
                f.write(f"\nகலாச்சார குறிப்புகள்:\n{story['cultural_notes']}\n\n")
        
        print(f"✓ Stories exported to {filename}")

def main():
    """Main function to demonstrate story generation"""
    print("="*60)
    print("Tamil Dialect Story Generator")
    print("="*60)
    
    # Initialize generator
    generator = TamilStoryGenerator()
    
    # Generate sample stories
    print("\n1. Generating a single story...")
    story = generator.generate_story(
        dialect="Kongu Tamil",
        theme="moral tale",
        length="medium"
    )
    
    print(f"\nGenerated Story:")
    print(f"Title: {story['title']}")
    print(f"Dialect: {story['dialect']}")
    print(f"Theme: {story['theme']}")
    print(f"\nContent:\n{story['content']}")
    print(f"\nMoral: {story['moral']}")
    
    # Generate batch of stories
    print("\n2. Generating batch of stories...")
    stories = generator.batch_generate_stories(count=5)
    
    # Save to files
    generator.save_stories(stories, "generated_stories.json")
    generator.export_to_txt(stories, "tamil_stories1.txt")
    
    print("\n" + "="*60)
    print("Story generation complete!")
    print("Files created:")
    print("  - generated_stories.json (JSON format)")
    print("  - tamil_stories1.txt (Text format)")
    print("="*60)

if __name__ == "__main__":
    main()