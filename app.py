from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import PyPDF2
import pandas as pd
import docx
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

MODEL_DIR = "./fine_tuned_gpt2"  
print("Loading fine-tuned GPT-2 model...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)

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

        try:
            if filename.endswith('.pdf'):
                text = extract_text_from_pdf(filepath)
            elif filename.endswith('.docx'):
                text = extract_text_from_docx(filepath)
            elif filename.endswith('.xlsx'):
                text = extract_text_from_excel(filepath)
            else:
                return jsonify({'error': 'Unsupported file format. Please upload PDF, DOCX, or XLSX files.'})

            if not text.strip():
                return jsonify({'error': 'Could not extract meaningful text from the uploaded file.'})

            summary = process_large_content(text)

            return jsonify({'summary': summary})

        except Exception as e:
            print(f"Error processing file: {e}")
            return jsonify({'error': 'Error processing the file.'})

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

    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    answer = generate_text(prompt, max_length=100)

    return jsonify({'answer': answer})

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """
    Endpoint to calculate and return model performance metrics.
    """
    try:
        # Example usage
        model_metrics = calculate_metrics(model, dataset, tokenizer)
        competitor_metrics = {
            "latency": 60.0,
            "accuracy": 88.0,
            "efficiency": 90.0,
        }
        comparison = compare_model_performance(model_metrics, competitor_metrics)

        return jsonify({
            "model_metrics": model_metrics,
            "comparison": comparison,
        })
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return jsonify({"error": "Failed to calculate metrics"}), 500

def extract_text_from_pdf(filepath):
    try:
        with open(filepath, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ''
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                # Split into chunks of 1024 tokens if the page is too large
                if len(tokenizer.encode(text)) > 1024:
                    break  # Stop at max token length
            return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(filepath):
    try:
        doc = docx.Document(filepath)
        text = ''
        for paragraph in doc.paragraphs:
            text += paragraph.text + '\n'
        return text
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

def extract_text_from_excel(filepath):
    try:
        df = pd.read_excel(filepath, engine='openpyxl')
        text = df.to_string(index=False, header=True)
        return text
    except Exception as e:
        print(f"Error extracting text from Excel: {e}")
        return ""

def generate_text(prompt, max_length=150):
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)

        if len(inputs["input_ids"][0]) == 0:
            return "Error: Input text is too short."

        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            max_length=max_length + len(inputs["input_ids"][0]),
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=50256,
            eos_token_id=50256,
            no_repeat_ngram_size=2,
        )

        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        print(f"Error generating text: {e}")
        return "Error in text generation."

def split_into_chunks(text, max_tokens=1024):
    """Split text into smaller chunks of a maximum number of tokens"""
    tokens = tokenizer.encode(text)
    # Split tokens into chunks if the token length exceeds max_tokens
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

def process_large_content(text):
    chunks = split_into_chunks(text, max_tokens=1024)
    summary = ""
    for chunk in chunks:
        summary += generate_text(f"Summarize this text:\n{chunk}", max_length=150) + "\n"
    return summary

if __name__ == '__main__':
    app.run(debug=True)
