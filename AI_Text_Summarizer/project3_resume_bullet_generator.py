"""
Project 3: AI Resume Bullet Generator
A practical tool that generates professional resume bullet points based on job descriptions and skills.
Perfect for job seekers, career changers, or anyone looking to improve their resume.

Required libraries:
- transformers
- torch
- re

Install with: pip install transformers torch
"""

import torch
from transformers import pipeline
import re
import textwrap

class ResumeBulletGenerator:
    def __init__(self):
        """Initialize the resume bullet generator with a pre-trained model."""
        print("Loading AI resume bullet generation model...")
        self.generator = pipeline(
            "text-generation",
            model="gpt2",  # Good for structured text generation
            device=-1  # Use CPU (-1) instead of GPU (0)
        )
        print("Model loaded successfully!")
    
    def clean_text(self, text):
        """Clean and format text for better generation."""
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s\-\.]', '', text)
        return text
    
    def generate_bullet_point(self, skill, context, max_length=50):
        """
        Generate a professional bullet point for a given skill and context.
        
        Args:
            skill (str): The skill or responsibility
            context (str): Additional context or industry
            max_length (int): Maximum length of the bullet point
        
        Returns:
            str: Generated bullet point
        """
        # Create a structured prompt
        prompt = f"Developed {skill} in {context} to"
        
        try:
            # Generate the bullet point
            result = self.generator(
                prompt,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.generator.tokenizer.eos_token_id,
                num_return_sequences=1
            )
            
            # Extract and clean the generated text
            generated_text = result[0]['generated_text']
            
            # Clean up the text
            bullet_point = self.clean_bullet_point(generated_text, prompt)
            
            return bullet_point
            
        except Exception as e:
            return f"Error generating bullet point: {str(e)}"
    
    def clean_bullet_point(self, text, original_prompt):
        """Clean and format the generated bullet point."""
        # Remove the original prompt from the beginning
        if text.startswith(original_prompt):
            text = text[len(original_prompt):]
        
        # Clean up the text
        text = text.strip()
        
        # Ensure it ends with a period
        if text and not text.endswith('.'):
            text += '.'
        
        # Combine with the original prompt
        full_bullet = original_prompt + text
        
        # Clean up any extra whitespace
        full_bullet = ' '.join(full_bullet.split())
        
        return full_bullet
    
    def generate_multiple_bullets(self, skills, context, num_bullets=3):
        """
        Generate multiple bullet points for a set of skills.
        
        Args:
            skills (list): List of skills or responsibilities
            context (str): Industry or company context
            num_bullets (int): Number of bullet points to generate per skill
        
        Returns:
            dict: Dictionary with skills as keys and lists of bullet points as values
        """
        results = {}
        
        for skill in skills:
            skill = skill.strip()
            if skill:
                bullets = []
                for i in range(num_bullets):
                    bullet = self.generate_bullet_point(skill, context)
                    bullets.append(bullet)
                results[skill] = bullets
        
        return results
    
    def format_output(self, skill, bullets):
        """Format the output for better readability."""
        print(f"\n🎯 Skill: {skill}")
        print("-" * 50)
        for i, bullet in enumerate(bullets, 1):
            print(f"{i}. {bullet}")
    
    def get_skill_suggestions(self):
        """Provide common skill suggestions for different industries."""
        suggestions = {
            "Software Development": [
                "Python programming",
                "Web development",
                "Database management",
                "API development",
                "Version control",
                "Testing and debugging"
            ],
            "Data Science": [
                "Data analysis",
                "Machine learning",
                "Statistical modeling",
                "Data visualization",
                "SQL programming",
                "Predictive analytics"
            ],
            "Marketing": [
                "Digital marketing",
                "Social media management",
                "Content creation",
                "SEO optimization",
                "Email marketing",
                "Campaign management"
            ],
            "Sales": [
                "Customer relationship management",
                "Lead generation",
                "Sales presentations",
                "Negotiation",
                "Market research",
                "Account management"
            ],
            "Project Management": [
                "Team leadership",
                "Agile methodology",
                "Risk management",
                "Budget planning",
                "Stakeholder communication",
                "Resource allocation"
            ]
        }
        
        print("\n💼 Industry Skill Suggestions:")
        print("-" * 40)
        for industry, skills in suggestions.items():
            print(f"\n{industry}:")
            for skill in skills:
                print(f"  • {skill}")
        
        return suggestions

def main():
    """Main function to run the resume bullet generator."""
    print("📄 AI Resume Bullet Generator")
    print("=" * 40)
    
    # Initialize the generator
    generator = ResumeBulletGenerator()
    
    while True:
        print("\nChoose an option:")
        print("1. Generate bullet points for specific skills")
        print("2. Use industry skill suggestions")
        print("3. Generate single bullet point")
        print("4. View skill suggestions")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            print("\nEnter your skills (comma-separated):")
            skills_input = input("> ").strip()
            context = input("Enter industry/company context: ").strip()
            
            if skills_input and context:
                skills = [skill.strip() for skill in skills_input.split(',')]
                print(f"\nGenerating bullet points for {len(skills)} skills...")
                
                results = generator.generate_multiple_bullets(skills, context)
                
                print("\n" + "="*60)
                print("📄 GENERATED RESUME BULLET POINTS")
                print("="*60)
                
                for skill, bullets in results.items():
                    generator.format_output(skill, bullets)
                
                print("="*60)
            else:
                print("Please enter both skills and context.")
        
        elif choice == "2":
            suggestions = generator.get_skill_suggestions()
            print("\nEnter industry name from suggestions above:")
            industry = input("> ").strip()
            
            if industry in suggestions:
                skills = suggestions[industry]
                context = input("Enter specific company/role context: ").strip()
                
                if context:
                    print(f"\nGenerating bullet points for {industry} skills...")
                    results = generator.generate_multiple_bullets(skills, context)
                    
                    print("\n" + "="*60)
                    print(f"📄 GENERATED BULLET POINTS FOR {industry.upper()}")
                    print("="*60)
                    
                    for skill, bullets in results.items():
                        generator.format_output(skill, bullets)
                    
                    print("="*60)
                else:
                    print("Please enter a context.")
            else:
                print("Industry not found. Please check the spelling.")
        
        elif choice == "3":
            skill = input("\nEnter skill: ").strip()
            context = input("Enter context: ").strip()
            
            if skill and context:
                print("\nGenerating bullet point...")
                bullet = generator.generate_bullet_point(skill, context)
                
                print("\n" + "="*60)
                print("📄 GENERATED BULLET POINT")
                print("="*60)
                print(f"Skill: {skill}")
                print(f"Context: {context}")
                print("-" * 40)
                print(f"• {bullet}")
                print("="*60)
            else:
                print("Please enter both skill and context.")
        
        elif choice == "4":
            generator.get_skill_suggestions()
        
        elif choice == "5":
            print("Thank you for using the AI Resume Bullet Generator!")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")
        
        # Ask if user wants to continue
        if choice in ["1", "2", "3"]:
            continue_choice = input("\nGenerate more bullet points? (y/n): ").strip().lower()
            if continue_choice != 'y':
                print("Thank you for using the AI Resume Bullet Generator!")
                break

if __name__ == "__main__":
    main() 