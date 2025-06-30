"""
Project 2: AI Story Generator
A creative tool that generates engaging stories based on user prompts using HuggingFace's text generation models.
Perfect for writers, content creators, or anyone looking for creative inspiration.

Required libraries:
- transformers
- torch
- random

Install with: pip install transformers torch
"""

import torch
from transformers import pipeline, set_seed
import random
import textwrap
import time

class StoryGenerator:
    def __init__(self):
        """Initialize the story generator with a pre-trained model."""
        print("Loading AI story generation model...")
        self.generator = pipeline(
            "text-generation",
            model="gpt2",  # Good for creative text generation
            device=-1  # Use CPU (-1) instead of GPU (0)
        )
        print("Model loaded successfully!")
        
        # Set a random seed for reproducible results
        set_seed(random.randint(1, 1000))
    
    def generate_story(self, prompt, max_length=200, temperature=0.8):
        """
        Generate a story based on the given prompt.
        
        Args:
            prompt (str): The story prompt or beginning
            max_length (int): Maximum length of generated text
            temperature (float): Controls randomness (0.1 = focused, 1.0 = creative)
        
        Returns:
            str: Generated story
        """
        try:
            # Generate the story
            result = self.generator(
                prompt,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.generator.tokenizer.eos_token_id,
                num_return_sequences=1
            )
            
            # Extract the generated text
            generated_text = result[0]['generated_text']
            
            # Clean up the text
            story = self.clean_text(generated_text)
            
            return story
            
        except Exception as e:
            return f"Error generating story: {str(e)}"
    
    def clean_text(self, text):
        """Clean and format the generated text."""
        # Remove any incomplete sentences at the end
        sentences = text.split('.')
        if len(sentences) > 1:
            # Keep all complete sentences
            complete_sentences = sentences[:-1]
            text = '. '.join(complete_sentences) + '.'
        
        # Clean up extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def create_story_prompt(self, genre, character, setting, conflict):
        """Create a structured story prompt based on user inputs."""
        prompts = {
            "adventure": f"In a {setting}, {character} embarks on an epic journey to {conflict}. ",
            "mystery": f"When {character} discovers {conflict} in {setting}, they must solve the mystery before it's too late. ",
            "romance": f"Amidst the {setting}, {character} finds themselves caught in a whirlwind romance while dealing with {conflict}. ",
            "scifi": f"In the year 2157, {character} navigates through {setting} to prevent {conflict} from destroying humanity. ",
            "fantasy": f"In the mystical realm of {setting}, {character} must harness ancient powers to overcome {conflict}. ",
            "horror": f"Deep within {setting}, {character} encounters {conflict} that threatens to consume their very soul. "
        }
        
        return prompts.get(genre.lower(), f"{character} finds themselves in {setting} facing {conflict}. ")
    
    def format_output(self, prompt, story):
        """Format the output for better readability."""
        print("\n" + "="*60)
        print("📝 STORY PROMPT")
        print("="*60)
        print(textwrap.fill(prompt, width=80))
        
        print("\n" + "="*60)
        print("✨ AI GENERATED STORY")
        print("="*60)
        print(textwrap.fill(story, width=80))
        print("="*60)
    
    def get_story_suggestions(self):
        """Provide story prompt suggestions to help users get started."""
        suggestions = [
            {
                "genre": "Adventure",
                "character": "a young explorer",
                "setting": "ancient ruins",
                "conflict": "find a lost treasure"
            },
            {
                "genre": "Mystery",
                "character": "a detective",
                "setting": "a small coastal town",
                "conflict": "a series of unexplained disappearances"
            },
            {
                "genre": "Sci-Fi",
                "character": "a space pilot",
                "setting": "distant galaxy",
                "conflict": "save Earth from an alien invasion"
            },
            {
                "genre": "Fantasy",
                "character": "a young wizard",
                "setting": "magical academy",
                "conflict": "master forbidden spells"
            }
        ]
        
        print("\n📚 Story Prompt Suggestions:")
        print("-" * 40)
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion['genre']}: {suggestion['character']} in {suggestion['setting']} must {suggestion['conflict']}")
        
        return suggestions

def main():
    """Main function to run the story generator."""
    print("✨ AI Story Generator")
    print("=" * 40)
    
    # Initialize the story generator
    generator = StoryGenerator()
    
    while True:
        print("\nChoose an option:")
        print("1. Generate story with custom prompt")
        print("2. Use guided story creation")
        print("3. Use a random story suggestion")
        print("4. View story suggestions")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            print("\nEnter your story prompt:")
            prompt = input("> ").strip()
            if prompt:
                print("\nGenerating your story...")
                story = generator.generate_story(prompt)
                generator.format_output(prompt, story)
            else:
                print("Please enter a valid prompt.")
        
        elif choice == "2":
            print("\nLet's create a story together!")
            genre = input("Enter genre (adventure/mystery/romance/scifi/fantasy/horror): ").strip()
            character = input("Enter main character: ").strip()
            setting = input("Enter setting: ").strip()
            conflict = input("Enter main conflict/challenge: ").strip()
            
            if all([genre, character, setting, conflict]):
                prompt = generator.create_story_prompt(genre, character, setting, conflict)
                print(f"\nGenerated prompt: {prompt}")
                print("\nGenerating your story...")
                story = generator.generate_story(prompt)
                generator.format_output(prompt, story)
            else:
                print("Please fill in all fields.")
        
        elif choice == "3":
            suggestions = generator.get_story_suggestions()
            suggestion = random.choice(suggestions)
            prompt = generator.create_story_prompt(
                suggestion['genre'],
                suggestion['character'],
                suggestion['setting'],
                suggestion['conflict']
            )
            print(f"\nUsing random suggestion: {prompt}")
            print("\nGenerating your story...")
            story = generator.generate_story(prompt)
            generator.format_output(prompt, story)
        
        elif choice == "4":
            generator.get_story_suggestions()
        
        elif choice == "5":
            print("Thank you for using the AI Story Generator!")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")
        
        # Ask if user wants to continue
        if choice in ["1", "2", "3"]:
            continue_choice = input("\nGenerate another story? (y/n): ").strip().lower()
            if continue_choice != 'y':
                print("Thank you for using the AI Story Generator!")
                break

if __name__ == "__main__":
    main() 