"""
Project 1: AI Text Summarizer
A practical tool that uses HuggingFace's transformers to summarize long text documents.
Perfect for students, researchers, or anyone who needs to quickly understand long articles.

Required libraries:
- transformers
- torch
- nltk
- flask

Install with: pip install transformers torch nltk flask
"""

import torch
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
import textwrap

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextSummarizer:
    def __init__(self):
        """Initialize the summarizer with a pre-trained model."""
        print("Loading AI summarization model...")
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",  # Good for news/article summarization
            device=-1  # Use CPU (-1) instead of GPU (0)
        )
        print("Model loaded successfully!")
    
    def preprocess_text(self, text):
        """Clean and prepare text for summarization."""
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        return text
    
    def split_long_text(self, text, max_length=1024):
        """Split very long text into chunks that the model can handle."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < max_length:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def summarize_text(self, text, max_length=130, min_length=30):
        """
        Summarize the given text.
        
        Args:
            text (str): Input text to summarize
            max_length (int): Maximum length of summary
            min_length (int): Minimum length of summary
        
        Returns:
            str: Generated summary
        """
        # Preprocess the text
        text = self.preprocess_text(text)
        
        # Check if text is too short to summarize
        if len(text.split()) < 50:
            return "Text is too short to summarize meaningfully."
        
        # Split long text into chunks
        if len(text) > 1024:
            print("Text is very long, processing in chunks...")
            chunks = self.split_long_text(text)
            summaries = []
            
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)}...")
                summary = self.summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
                summaries.append(summary[0]['summary_text'])
            
            # Combine summaries
            combined_summary = " ".join(summaries)
            # Summarize the combined summary if it's still too long
            if len(combined_summary.split()) > max_length:
                final_summary = self.summarizer(combined_summary, max_length=max_length, min_length=min_length, do_sample=False)
                return final_summary[0]['summary_text']
            return combined_summary
        else:
            # Process normally for shorter text
            summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text']
    
    def format_output(self, original_text, summary):
        """Format the output for better readability."""
        print("\n" + "="*60)
        print("📄 ORIGINAL TEXT")
        print("="*60)
        print(f"Length: {len(original_text.split())} words")
        print("-" * 40)
        print(textwrap.fill(original_text, width=80))
        
        print("\n" + "="*60)
        print("🤖 AI GENERATED SUMMARY")
        print("="*60)
        print(f"Length: {len(summary.split())} words")
        print("-" * 40)
        print(textwrap.fill(summary, width=80))
        print("="*60)

def main():
    """Main function to run the text summarizer."""
    print("🤖 AI Text Summarizer")
    print("=" * 40)
    
    # Initialize the summarizer
    summarizer = TextSummarizer()
    
    # Example texts for demonstration
    sample_texts = {
        "1": """Artificial Intelligence (AI) has emerged as one of the most transformative technologies of the 21st century. 
        From virtual assistants like Siri and Alexa to recommendation systems on Netflix and Amazon, AI is becoming increasingly 
        integrated into our daily lives. Machine learning, a subset of AI, enables computers to learn and improve from experience 
        without being explicitly programmed. Deep learning, which uses neural networks with multiple layers, has revolutionized 
        fields like computer vision, natural language processing, and speech recognition. Companies are investing billions in AI 
        research and development, recognizing its potential to drive innovation and create competitive advantages. However, the 
        rise of AI also raises important questions about job displacement, privacy, and ethical considerations. As AI continues 
        to evolve, it's crucial to develop frameworks that ensure responsible and beneficial use of this powerful technology.""",
        
        "2": """Climate change represents one of the most pressing challenges facing humanity today. The Earth's average surface 
        temperature has increased by about 1.1 degrees Celsius since the pre-industrial era, primarily due to human activities 
        that release greenhouse gases into the atmosphere. The burning of fossil fuels for energy production, transportation, 
        and industrial processes is the largest contributor to these emissions. Deforestation and agricultural practices also 
        play significant roles. The consequences of climate change are already visible worldwide: rising sea levels, more frequent 
        and severe weather events, melting glaciers, and shifts in plant and animal habitats. These changes threaten food security, 
        water availability, and human health. Addressing climate change requires global cooperation and immediate action to reduce 
        emissions, transition to renewable energy sources, and develop sustainable practices across all sectors of society."""
    }
    
    while True:
        print("\nChoose an option:")
        print("1. Use sample text about AI")
        print("2. Use sample text about Climate Change")
        print("3. Enter your own text")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "4":
            print("Thank you for using the AI Text Summarizer!")
            break
        elif choice in ["1", "2"]:
            text = sample_texts[choice]
            summary = summarizer.summarize_text(text)
            summarizer.format_output(text, summary)
        elif choice == "3":
            print("\nEnter your text (press Enter twice to finish):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            
            if lines:
                text = " ".join(lines)
                summary = summarizer.summarize_text(text)
                summarizer.format_output(text, summary)
            else:
                print("No text entered. Please try again.")
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main() 