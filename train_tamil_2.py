"""
Tamil Dialect Model Training with LoRA
Fine-tune language models for Tamil dialect processing
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json
import os
from datetime import datetime
import numpy as np
import argparse

class TamilDialectTrainer:
    def __init__(self, base_model="gpt2", output_dir="./tamil_model"):
        """
        Initialize trainer for Tamil dialect models
        
        In production, use:
        - "ai4bharat/IndicBART" for Indian languages
        - "meta-llama/Llama-2-7b-hf" for larger models
        - Custom Tamil ASR models
        """
        self.base_model = base_model
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Initializing Tamil Dialect Trainer")
        print(f"Base Model: {base_model}")
        print(f"Device: {self.device}")
        print(f"Output Directory: {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
    
    def load_model_and_tokenizer(self):
        """Load base model and tokenizer"""
        print("\nLoading model and tokenizer...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            print("✓ Model and tokenizer loaded successfully!")
            return True
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    
    def configure_lora(self, r=8, lora_alpha=16, lora_dropout=0.05):
        """
        Configure LoRA (Low-Rank Adaptation) for efficient fine-tuning
        
        LoRA reduces trainable parameters significantly while maintaining performance
        """
        print("\nConfiguring LoRA...")
        
        lora_config = LoraConfig(
            r=r,  # Rank of the update matrices
            lora_alpha=lora_alpha,  # Scaling factor
            target_modules=["c_attn", "c_proj"],  # Modules to apply LoRA (model-specific)
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"✓ LoRA configured successfully!")
        print(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        print(f"  Total parameters: {total_params:,}")
        
        return lora_config
    
    def prepare_dataset(self, stories_file="generated_stories.json"):
        """Prepare dataset from Tamil stories"""
        print(f"\nPreparing dataset from {stories_file}...")
        
        # Load stories
        try:
            with open(stories_file, 'r', encoding='utf-8') as f:
                stories = json.load(f)
        except FileNotFoundError:
            print(f"✗ File not found. Generating sample dataset...")
            stories = self._generate_sample_stories()
        
        # Prepare training data
        texts = []
        for story in stories:
            # Format: [Dialect] Title\n\nContent\n\nMoral: moral\nProverbs: proverbs
            text = f"[{story['dialect']}] {story['title']}\n\n"
            text += f"{story['content']}\n\n"
            text += f"பாடம்: {story['moral']}\n"
            text += f"பழமொழிகள்: {', '.join(story['proverbs'])}"
            texts.append(text)
        
        # Create dataset
        dataset = Dataset.from_dict({"text": texts})
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Split into train and validation
        split = tokenized_dataset.train_test_split(test_size=0.1)
        
        print(f"✓ Dataset prepared!")
        print(f"  Training samples: {len(split['train'])}")
        print(f"  Validation samples: {len(split['test'])}")
        
        return split['train'], split['test']
    
    def _generate_sample_stories(self):
        """Generate sample stories for demonstration"""
        return [
            {
                "title": "முயற்சியின் வெற்றி",
                "dialect": "Kongu Tamil",
                "content": "ஒரு காலத்தில் ஒரு விவசாயி கடுமையாக உழைத்தார். அவரது முயற்சி வெற்றி பெற்றது.",
                "moral": "விடாமுயற்சி வெற்றியைத் தரும்",
                "proverbs": ["முயற்சி திருவினையாக்கும்"]
            },
            {
                "title": "புத்திசாலி நரி",
                "dialect": "Madurai Tamil",
                "content": "காட்டில் ஒரு நரி வாழ்ந்தது. அது புத்திசாலித்தனமாக செயல்பட்டது.",
                "moral": "புத்தி பலத்தை வெல்லும்",
                "proverbs": ["அறிவுடையார் எல்லாம் உடையார்"]
            }
        ] * 20  # Duplicate for more training data
    
    def train(self, train_dataset, eval_dataset, epochs=3, batch_size=4, learning_rate=2e-4):
        """Train the model with prepared dataset"""
        print("\nStarting training...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            push_to_hub=False,
            report_to="none",
            fp16=torch.cuda.is_available(),
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Train
        print("\nTraining in progress...")
        print("=" * 60)
        
        try:
            train_result = trainer.train()
            
            print("=" * 60)
            print("✓ Training completed!")
            print(f"  Training loss: {train_result.training_loss:.4f}")
            print(f"  Training time: {train_result.metrics['train_runtime']:.2f}s")
            
            # Evaluate
            eval_results = trainer.evaluate()
            print(f"\nEvaluation Results:")
            print(f"  Eval loss: {eval_results['eval_loss']:.4f}")
            
            # Save model
            trainer.save_model(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            
            print(f"\n✓ Model saved to {self.output_dir}")
            
            return trainer, train_result, eval_results
            
        except Exception as e:
            print(f"✗ Training error: {e}")
            return None, None, None
    
    def test_generation(self, prompt="ஒரு காலத்தில்", max_length=100):
        """Test the fine-tuned model"""
        print("\nTesting model generation...")
        print(f"Prompt: {prompt}")
        print("-" * 60)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                num_return_sequences=1
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nGenerated:\n{generated_text}")
        print("-" * 60)
        
        return generated_text
    
    def save_training_info(self, config):
        """Save training configuration and results"""
        info = {
            "base_model": self.base_model,
            "training_date": datetime.now().isoformat(),
            "device": str(self.device),
            "config": config,
            "output_dir": self.output_dir
        }
        
        with open(os.path.join(self.output_dir, "training_info.json"), 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"✓ Training info saved")

def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description="Tamil dialect training pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Run checks only (load model/tokenizer, prepare dataset) and skip training")
    args = parser.parse_args()
    print("="*60)
    print("Tamil Dialect Model Training with LoRA")
    print("="*60)
    
    # Initialize trainer
    trainer = TamilDialectTrainer(
        base_model="gpt2",  # Change to "ai4bharat/IndicBART" for better Tamil support
        output_dir="./tamil_dialect_model"
    )
    
    # Load model
    if not trainer.load_model_and_tokenizer():
        print("Failed to load model. Exiting...")
        return
    
    # Configure LoRA
    lora_config = trainer.configure_lora(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05
    )
    
    # Prepare dataset
    train_dataset, eval_dataset = trainer.prepare_dataset("generated_stories.json")

    if args.dry_run:
        print("\nDry-run enabled: skipped training and saving. Exiting after successful load and dataset preparation.")
        return
    
    # Train model
    model_trainer, train_result, eval_results = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        epochs=3,
        batch_size=4,
        learning_rate=2e-4
    )
    
    if model_trainer is not None:
        # Test generation
        test_prompts = [
            "ஒரு காலத்தில் ",
            "[Kongu Tamil] ",
            "காட்டில் ஒரு "
        ]
        
        print("\n" + "="*60)
        print("Testing Model Generations")
        print("="*60)
        
        for prompt in test_prompts:
            trainer.test_generation(prompt, max_length=150)
        
        # Save training info
        trainer.save_training_info({
            "lora_r": 8,
            "lora_alpha": 16,
            "epochs": 3,
            "batch_size": 4,
            "learning_rate": 2e-4
        })
        
        print("\n" + "="*60)
        print("Training Pipeline Complete!")
        print("="*60)
        print(f"\nModel saved to: ./tamil_dialect_model")
        print("You can now use this model for:")
        print("  - Dialect-specific story generation")
        print("  - Cultural content creation")
        print("  - Language preservation tasks")
        print("="*60)
    else:
        print("\n✗ Training failed. Please check the errors above.")

if __name__ == "__main__":
    main()