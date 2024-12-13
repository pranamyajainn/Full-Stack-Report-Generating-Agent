import os
import json
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import pytesseract
import openai
from dotenv import load_dotenv
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage, ListFlowable, ListItem
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import seaborn as sns
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'csv', 'xlsx', 'json', 'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# OpenAI API configuration
openai.api_key = os.getenv("OPENAI_API_KEY")

# Utility functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_image(file_path):
    try:
        image = PILImage.open(file_path)
        text = pytesseract.image_to_string(image)
        return text.strip() or "No text could be extracted from the image."
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"

def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        extracted_text = ''.join(page.extract_text() for page in reader.pages)
        return extracted_text.strip() or "No text could be extracted from the PDF."
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

def process_file(file_path, file_extension):
    try:
        if file_extension in {"csv", "xlsx"}:
            df = pd.read_csv(file_path) if file_extension == "csv" else pd.read_excel(file_path)
            return df.to_string(index=False), df
        elif file_extension == "pdf":
            return extract_text_from_pdf(file_path), None
        elif file_extension in {"jpg", "jpeg", "png"}:
            return extract_text_from_image(file_path), None
        elif file_extension == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read(), None
        elif file_extension == "json":
            with open(file_path, "r", encoding="utf-8") as f:
                return json.dumps(json.load(f), indent=4), None
        else:
            return f"Unsupported file type: {file_extension}", None
    except Exception as e:
        return f"Error processing file {file_path}: {str(e)}", None

def retrieve_relevant_chunks(prompt, report_type, file_content):
    """
    Retrieve relevant information using Llama Index.
    """
    try:
        Settings.llm = OpenAI(model="gpt-3.5-turbo")

        # Save file content temporarily for Llama Index processing
        temp_dir = os.path.join(UPLOAD_FOLDER, 'temp_docs')
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, 'temp_document.txt')
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            f.write(file_content)

        # Load documents and create Llama Index
        documents = SimpleDirectoryReader(temp_dir).load_data()
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        
        # Predefined system prompts
        system_prompts = {
        "general": "You are a general-purpose report generator. Your job is to generate structured, professional reports.",
        "financial": "You are a high professional financial report generator. Your job is to generate structured, professional financial reports.",
        "stock": "You are a high professional stock report generator. Your job is to generate structured, professional stock reports.",
        "student_exam": "You are a high professional student exam report generator. Your job is to generate structured, professional student exam reports.",
        "research": "You are a high professional research report generator. Your job is to generate structured, professional research reports.",
        "crime": "You are a high professional crime report generator. Your job is to generate structured, professional crime reports."
        }
        full_query = f"{system_prompts.get(report_type, 'Provide a comprehensive and professional summary.')}\n\n{prompt}"
        response = query_engine.query(full_query)
        return str(response)
    except Exception as e:
        print(f"Error in Llama Index retrieval: {e}")
        return "No relevant information could be retrieved."
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def generate_detailed_pdf_with_visualizations(prompt, combined_responses, file_summaries, df):
    pdf_path = os.path.join(STATIC_FOLDER, 'detailed_report.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle('TitleStyle', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=18, textColor=colors.HexColor("#003366"), spaceAfter=12)
    heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=14, textColor=colors.HexColor("#003366"), spaceAfter=10)
    body_style = ParagraphStyle('BodyStyle', parent=styles['BodyText'], fontName='Helvetica', fontSize=11, textColor=colors.black, leading=14, spaceAfter=8)

    elements = []

    # Title
    elements.append(Paragraph("AI-Report Generating Agent", title_style))
    elements.append(Spacer(1, 12))

    # Introduction
    elements.append(Paragraph("Introduction", heading_style))
    elements.append(Paragraph(f"This report was generated based on the prompt: '{prompt}'.", body_style))
    elements.append(Spacer(1, 12))

    # Findings
    elements.append(Paragraph("Findings", heading_style))
    for response in combined_responses:
        bullet_items = [ListItem(Paragraph(line.strip(), body_style)) for line in response.split('\n') if line.strip()]
        elements.append(ListFlowable(bullet_items, bulletType='bullet'))
        elements.append(Spacer(1, 12))

    # Data Visualizations
    if not df.empty:
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            try:
                elements.append(Paragraph("Data Visualizations", heading_style))

                # Bar Chart
                bar_chart_path = os.path.join(STATIC_FOLDER, 'bar_chart.png')
                numeric_df.plot(kind='bar', figsize=(8, 6))
                plt.title("Bar Chart")
                plt.tight_layout()
                plt.savefig(bar_chart_path)
                plt.close()
                elements.append(ReportLabImage(bar_chart_path, width=400, height=300))
                elements.append(Spacer(1, 12))

                # Pie Chart
                pie_chart_path = os.path.join(STATIC_FOLDER, 'pie_chart.png')
                numeric_df.sum().plot(kind='pie', autopct='%1.1f%%', figsize=(6, 6))
                plt.title("Pie Chart")
                plt.tight_layout()
                plt.savefig(pie_chart_path)
                plt.close()
                elements.append(ReportLabImage(pie_chart_path, width=300, height=300))
                elements.append(Spacer(1, 12))

                # Heat Map
                heatmap_path = os.path.join(STATIC_FOLDER, 'heatmap.png')
                sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
                plt.title("Heatmap")
                plt.tight_layout()
                plt.savefig(heatmap_path)
                plt.close()
                elements.append(ReportLabImage(heatmap_path, width=400, height=300))
                elements.append(Spacer(1, 12))

            except Exception as e:
                elements.append(Paragraph(f"Error generating visualizations: {str(e)}", body_style))

    doc.build(elements)
    return pdf_path

@app.route('/preview', methods=['POST'])
def preview_report():
    if 'files' not in request.files or len(request.files.getlist('files')) == 0:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist('files')
    prompt = request.form.get('prompt', '')
    report_type = request.form.get('report_type', 'general')
    combined_responses, file_summaries, dfs = [], [], []

    for file in files:
        if allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(file_path)
            file_content, df = process_file(file_path, file.filename.rsplit(".", 1)[1].lower())
            file_summaries.append(file_content[:500])
            if df is not None:
                dfs.append(df)
            combined_responses.append(retrieve_relevant_chunks(prompt, report_type, file_content))

    try:
        combined_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        pdf_path = generate_detailed_pdf_with_visualizations(prompt, combined_responses, file_summaries, combined_df)
        return jsonify({"preview_url": f"/static/{os.path.basename(pdf_path)}"})
    except Exception as e:
        return jsonify({"error": f"Error generating PDF: {str(e)}"}), 500

@app.route('/')
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
