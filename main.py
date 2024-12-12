import os
import json
import openai
import pandas as pd
import fitz  # PyMuPDF
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from fpdf import FPDF
import traceback

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads'
PREVIEW_FOLDER = 'previews'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'csv', 'xlsx', 'json'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREVIEW_FOLDER'] = PREVIEW_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Ensure required directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREVIEW_FOLDER, exist_ok=True)

# OpenAI API Configuration
openai.api_key = os.getenv("OPENAI_API_KEY")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_files(files):
    combined_text = ""
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            file_extension = filename.rsplit('.', 1)[1].lower()
            try:
                if file_extension == "txt":
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                elif file_extension == "pdf":
                    doc = fitz.open(file_path)
                    content = "".join(page.get_text() for page in doc)
                elif file_extension == "csv":
                    df = pd.read_csv(file_path)
                    content = df.to_string(index=False)
                elif file_extension == "xlsx":
                    df = pd.read_excel(file_path)
                    content = df.to_string(index=False)
                elif file_extension == "json":
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = json.dumps(json.load(f), indent=4)
                else:
                    content = ""

                combined_text += content + "\n"
            except Exception as e:
                return f"Error processing file {filename}: {str(e)}", False
    return combined_text, True


def generate_pdf(output_filename, title, content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=title, ln=True, align="C")
    pdf.multi_cell(0, 10, txt=content)
    pdf.output(output_filename)


def use_openai_chat_api(prompt, content, model="gpt-3.5-turbo", max_tokens=2000):
    """
    Function to query OpenAI API for generating a structured report.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "You are a report generator. Your job is to generate structured, professional reports "
                    "based on the provided prompt and content. Format your response clearly with headings, "
                    "bullet points, and sections as appropriate."
                )},
                {"role": "user", "content": (
                    f"Generate a detailed report using the following:\n\n"
                    f"Prompt: {prompt}\n\n"
                    f"Content: {content}"
                )}
            ],
            max_tokens=max_tokens
        )
        # Extract and return the AI response
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"OpenAI API Error: {str(e)}")
        return f"OpenAI API Error: {str(e)}"


@app.route('/preview', methods=['POST'])
def preview_report():
    try:
        if 'files' not in request.files:
            raise ValueError("No files uploaded")
        files = request.files.getlist('files')
        prompt = request.form.get('prompt', '')

        # Process files
        file_content, success = process_files(files)
        if not success:
            raise ValueError(file_content)

        # Call OpenAI API with the prompt and file content
        openai_response = use_openai_chat_api(prompt, file_content)

        # Generate PDF
        output_filename = os.path.join(app.config['PREVIEW_FOLDER'], 'preview.pdf')
        generate_pdf(output_filename, "AI Report", f"Prompt:\n{prompt}\n\nFile Content:\n{file_content}\n\nAI Response:\n{openai_response}")

        return jsonify({"preview_url": f"/previews/preview.pdf"})

    except Exception as e:
        app.logger.error(f"Error in /preview: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/previews/<filename>')
def serve_preview(filename):
    return send_file(os.path.join(app.config['PREVIEW_FOLDER'], filename))


@app.route('/')
def index():
    return render_template('upload.html')


if __name__ == "__main__":
    app.run(debug=True)