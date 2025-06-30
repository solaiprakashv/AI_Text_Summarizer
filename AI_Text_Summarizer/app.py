"""
Flask Web Application: AI Text Summarizer
A web-based version of the AI Text Summarizer using Flask and Bootstrap.

Required packages:
pip install flask transformers torch nltk

How to run:
1. Install dependencies: pip install flask transformers torch nltk
2. Run the app: python app.py
3. Open browser: http://localhost:5000

The app will automatically download the required AI model on first use.
"""

from flask import Flask, render_template_string, request, jsonify
import torch
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
import time

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize Flask app
app = Flask(__name__)

# Global variable to store the summarizer model
summarizer_model = None

def load_summarizer():
    """Load the AI summarization model."""
    global summarizer_model
    if summarizer_model is None:
        print("Loading AI summarization model...")
        summarizer_model = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1
        )
        print("Model loaded successfully!")
    return summarizer_model

def summarize_text(text, max_length=130, min_length=30):
    """Summarize the given text using the AI model."""
    try:
        summarizer = load_summarizer()
        text = ' '.join(text.split())
        
        if len(text.split()) < 50:
            return {
                'summary': 'Text is too short to summarize meaningfully.',
                'original_length': len(text.split()),
                'summary_length': 0,
                'processing_time': 0
            }
        
        start_time = time.time()
        
        if len(text) > 1024:
            sentences = sent_tokenize(text)
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk + sentence) < 1024:
                    current_chunk += sentence + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            summaries = []
            for chunk in chunks:
                summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
                summaries.append(summary[0]['summary_text'])
            
            combined_summary = " ".join(summaries)
            if len(combined_summary.split()) > max_length:
                final_summary = summarizer(combined_summary, max_length=max_length, min_length=min_length, do_sample=False)
                summary_text = final_summary[0]['summary_text']
            else:
                summary_text = combined_summary
        else:
            summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            summary_text = summary[0]['summary_text']
        
        processing_time = round(time.time() - start_time, 2)
        
        return {
            'summary': summary_text,
            'original_length': len(text.split()),
            'summary_length': len(summary_text.split()),
            'processing_time': processing_time
        }
        
    except Exception as e:
        return {
            'summary': f'Error: {str(e)}',
            'original_length': len(text.split()) if text else 0,
            'summary_length': 0,
            'processing_time': 0
        }

# HTML template with Bootstrap styling
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Text Summarizer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 20px 20px 0 0;
        }
        .form-control, .btn {
            border-radius: 10px;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            transition: all 0.3s ease;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
        .result-card {
            border: none;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }
        .loading {
            display: none;
        }
        .stats-badge {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .sample-texts {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
        }
        .sample-btn {
            margin: 5px;
            border-radius: 20px;
        }
    </style>
</head>
<body>
    <div class="container py-5">
        <div class="row justify-content-center">
            <div class="col-lg-10">
                <div class="main-container">
                    <!-- Header -->
                    <div class="header p-4 text-center">
                        <h1><i class="fas fa-robot me-2"></i>🤖 AI Text Summarizer</h1>
                        <p class="mb-0">Transform long articles into concise summaries using AI</p>
                    </div>
                    
                    <!-- Main Content -->
                    <div class="p-4">
                        <!-- Sample Texts Section -->
                        <div class="sample-texts mb-4">
                            <h5><i class="fas fa-lightbulb me-2"></i>Try Sample Texts</h5>
                            <button class="btn btn-outline-primary btn-sm sample-btn" onclick="loadSampleText('ai')">
                                <i class="fas fa-microchip me-1"></i>AI Article
                            </button>
                            <button class="btn btn-outline-primary btn-sm sample-btn" onclick="loadSampleText('climate')">
                                <i class="fas fa-leaf me-1"></i>Climate Change
                            </button>
                            <button class="btn btn-outline-primary btn-sm sample-btn" onclick="loadSampleText('tech')">
                                <i class="fas fa-laptop me-1"></i>Technology
                            </button>
                        </div>
                        
                        <!-- Input Form -->
                        <form id="summarizeForm" method="POST">
                            <div class="mb-3">
                                <label for="textInput" class="form-label">
                                    <i class="fas fa-file-text me-2"></i>Enter your text to summarize:
                                </label>
                                <textarea 
                                    class="form-control" 
                                    id="textInput" 
                                    name="text" 
                                    rows="8" 
                                    placeholder="Paste your article, document, or any long text here..."
                                    required
                                ></textarea>
                            </div>
                            
                            <div class="row mb-3">
                                <div class="col-md-6">
                                    <label for="maxLength" class="form-label">
                                        <i class="fas fa-ruler me-2"></i>Maximum Summary Length:
                                    </label>
                                    <input type="range" class="form-range" id="maxLength" name="max_length" 
                                           min="50" max="200" value="130" oninput="updateMaxLengthValue(this.value)">
                                    <div class="text-center">
                                        <span id="maxLengthValue" class="badge bg-primary">130 words</span>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <label for="minLength" class="form-label">
                                        <i class="fas fa-ruler-vertical me-2"></i>Minimum Summary Length:
                                    </label>
                                    <input type="range" class="form-range" id="minLength" name="min_length" 
                                           min="10" max="100" value="30" oninput="updateMinLengthValue(this.value)">
                                    <div class="text-center">
                                        <span id="minLengthValue" class="badge bg-primary">30 words</span>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="text-center">
                                <button type="submit" class="btn btn-primary btn-lg px-5">
                                    <i class="fas fa-magic me-2"></i>Generate Summary
                                </button>
                            </div>
                        </form>
                        
                        <!-- Loading Indicator -->
                        <div id="loading" class="loading text-center mt-4">
                            <div class="spinner-border text-primary" role="status">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                            <p class="mt-2 text-muted">AI is analyzing your text...</p>
                        </div>
                        
                        <!-- Results Section -->
                        <div id="results" class="mt-4" style="display: none;">
                            <div class="card result-card">
                                <div class="card-header bg-primary text-white">
                                    <h5 class="mb-0">
                                        <i class="fas fa-chart-line me-2"></i>Summary Statistics
                                    </h5>
                                </div>
                                <div class="card-body">
                                    <div class="row text-center">
                                        <div class="col-md-4">
                                            <div class="stats-badge p-2 rounded">
                                                <h6>Original Words</h6>
                                                <span id="originalWords">0</span>
                                            </div>
                                        </div>
                                        <div class="col-md-4">
                                            <div class="stats-badge p-2 rounded">
                                                <h6>Summary Words</h6>
                                                <span id="summaryWords">0</span>
                                            </div>
                                        </div>
                                        <div class="col-md-4">
                                            <div class="stats-badge p-2 rounded">
                                                <h6>Processing Time</h6>
                                                <span id="processingTime">0s</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="card result-card mt-3">
                                <div class="card-header bg-success text-white">
                                    <h5 class="mb-0">
                                        <i class="fas fa-robot me-2"></i>AI Generated Summary
                                    </h5>
                                </div>
                                <div class="card-body">
                                    <p id="summaryText" class="lead"></p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Sample texts
        const sampleTexts = {
            'ai': `Artificial Intelligence (AI) has emerged as one of the most transformative technologies of the 21st century. From virtual assistants like Siri and Alexa to recommendation systems on Netflix and Amazon, AI is becoming increasingly integrated into our daily lives. Machine learning, a subset of AI, enables computers to learn and improve from experience without being explicitly programmed. Deep learning, which uses neural networks with multiple layers, has revolutionized fields like computer vision, natural language processing, and speech recognition. Companies are investing billions in AI research and development, recognizing its potential to drive innovation and create competitive advantages. However, the rise of AI also raises important questions about job displacement, privacy, and ethical considerations. As AI continues to evolve, it's crucial to develop frameworks that ensure responsible and beneficial use of this powerful technology.`,
            
            'climate': `Climate change represents one of the most pressing challenges facing humanity today. The Earth's average surface temperature has increased by about 1.1 degrees Celsius since the pre-industrial era, primarily due to human activities that release greenhouse gases into the atmosphere. The burning of fossil fuels for energy production, transportation, and industrial processes is the largest contributor to these emissions. Deforestation and agricultural practices also play significant roles. The consequences of climate change are already visible worldwide: rising sea levels, more frequent and severe weather events, melting glaciers, and shifts in plant and animal habitats. These changes threaten food security, water availability, and human health. Addressing climate change requires global cooperation and immediate action to reduce emissions, transition to renewable energy sources, and develop sustainable practices across all sectors of society.`,
            
            'tech': `The rapid advancement of technology continues to reshape our world at an unprecedented pace. From the rise of cloud computing and edge computing to the proliferation of Internet of Things (IoT) devices, technology is becoming more interconnected and intelligent. Artificial intelligence and machine learning are driving automation across industries, while blockchain technology is revolutionizing how we think about trust and transactions. The development of 5G networks is enabling faster, more reliable connectivity, paving the way for innovations like autonomous vehicles and smart cities. Virtual and augmented reality are creating new ways to experience and interact with digital content. As we move toward a more digital future, cybersecurity becomes increasingly critical to protect our data and systems. The challenge lies in harnessing these technologies for positive impact while addressing concerns about privacy, security, and the digital divide.`
        };
        
        // Load sample text
        function loadSampleText(type) {
            document.getElementById('textInput').value = sampleTexts[type];
        }
        
        // Update range input values
        function updateMaxLengthValue(value) {
            document.getElementById('maxLengthValue').textContent = value + ' words';
        }
        
        function updateMinLengthValue(value) {
            document.getElementById('minLengthValue').textContent = value + ' words';
        }
        
        // Handle form submission
        document.getElementById('summarizeForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const text = formData.get('text');
            const maxLength = formData.get('max_length');
            const minLength = formData.get('min_length');
            
            if (!text.trim()) {
                alert('Please enter some text to summarize.');
                return;
            }
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            
            // Send request
            fetch('/summarize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    max_length: parseInt(maxLength),
                    min_length: parseInt(minLength)
                })
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading
                document.getElementById('loading').style.display = 'none';
                
                // Update results
                document.getElementById('originalWords').textContent = data.original_length;
                document.getElementById('summaryWords').textContent = data.summary_length;
                document.getElementById('processingTime').textContent = data.processing_time + 's';
                
                document.getElementById('summaryText').textContent = data.summary;
                
                // Show results
                document.getElementById('results').style.display = 'block';
                
                // Scroll to results
                document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                alert('Error: ' + error.message);
            });
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Main page route."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/summarize', methods=['POST'])
def summarize():
    """API endpoint for text summarization."""
    try:
        data = request.get_json()
        text = data.get('text', '')
        max_length = data.get('max_length', 130)
        min_length = data.get('min_length', 30)
        
        if not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        # Generate summary
        result = summarize_text(text, max_length, min_length)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting AI Text Summarizer Web App...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("⏳ The AI model will download automatically on first use...")
    app.run(debug=True, host='0.0.0.0', port=5000) 