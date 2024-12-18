from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import PyPDF2
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure uploads folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the fine-tuned GPT-2 model
MODEL_DIR = "./fine_tuned_gpt2"  # Folder where your fine-tuned GPT-2 model is saved
print("Loading fine-tuned GPT-2 model...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Extract text from PDF
        text = extract_text_from_pdf(filepath)
        if not text:
            return jsonify({'error': 'Could not extract text from the PDF.'})

        # Summarize the content using fine-tuned GPT-2
        summary = generate_text(f"Summarize this text:\n{text}", max_length=150)

        return jsonify({'summary': summary})

@app.route('/feedback', methods=["POST"])
def feedback():
    name = request.form.get("name")
    feedback = request.form.get("feedback")
    print(f"Feedback from {name}: {feedback}")
    return jsonify({"message": "Thank you for your feedback!"})

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question', '')
    context = data.get('context', '')

    if not question or not context:
        return jsonify({'error': 'Question or context is missing'})

    # Generate an answer using the fine-tuned GPT-2 model
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    answer = generate_text(prompt, max_length=100)

    return jsonify({'answer': answer})

def extract_text_from_pdf(filepath):
    """
    Extracts text from a PDF file.
    """
    try:
        with open(filepath, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None

def generate_text(prompt, max_length=150):
    """
    Generates text using the fine-tuned GPT-2 model.
    """
    try:
        # Tokenize the prompt and ensure padding is handled correctly
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)

        # Generate text with the model
        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),  # Make sure attention_mask is passed
            max_length=max_length + len(inputs["input_ids"][0]),  # To account for the input length
            temperature=0.7,  # Controls randomness
            top_p=0.9,        # Nucleus sampling
            top_k=50,         # Limits sampling pool
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=50256,  # Set pad token ID to EOS token ID
            eos_token_id=50256,  # Use the EOS token as the padding token
            no_repeat_ngram_size=2,  # Prevent repeating n-grams
        )

        # Decode and return the generated text
        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        print(f"Error generating text: {e}")
        return "Error in text generation."

if __name__ == '__main__':
    app.run(debug=True)
