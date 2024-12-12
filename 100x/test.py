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
import openai  # OpenAI API integration
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
import os
# Load environment variables
from dotenv import load_dotenv
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
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set your OpenAI API key in the .env file

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


def retrieve_relevant_chunks(prompt, report_type, file_content):
    """
    Generates a report by sending the prompt and file content to OpenAI.

    Args:
        prompt (str): The user's prompt.
        report_type (str): The type of report requested (e.g., financial, research).
        file_content (str): The extracted content from the uploaded file.

    Returns:
        str: The response from OpenAI's API.
    """
    system_messages = {
        "general": "You are a general-purpose report generator. Your job is to generate structured, professional reports.",
        "financial": "You are a high professional financial report generator. Your job is to generate structured, professional financial reports.",
        "stock": "You are a high professional stock report generator. Your job is to generate structured, professional stock reports.",
        "student_exam": "You are a high professional student exam report generator. Your job is to generate structured, professional student exam reports.",
        "research": "You are a high professional research report generator. Your job is to generate structured, professional research reports.",
        "crime": "You are a high professional crime report generator. Your job is to generate structured, professional crime reports."
    }

    system_message = system_messages.get(report_type, "You are a general-purpose report generator.")

    # Combine the prompt with the dataset content
    combined_prompt = f"{prompt}\n\nDataset:\n{file_content[:2000]}"  # Limit to 2000 characters to avoid token limits.

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": combined_prompt}
            ],
            max_tokens=1000
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return "No relevant information could be retrieved."

def process_file(file_path, file_extension):
    try:
        if file_extension in {"csv", "xlsx"}:
            df = pd.read_csv(file_path) if file_extension == "csv" else pd.read_excel(file_path)
            return df.to_string(index=False), df  # Return both the string and the DataFrame
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
from reportlab.lib.units import inch

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage, ListFlowable, ListItem
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import matplotlib.pyplot as plt
import seaborn as sns
import os


def generate_detailed_pdf_with_visualizations(prompt, combined_responses, file_summaries, df):
    pdf_path = os.path.join(STATIC_FOLDER, 'detailed_report.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=18,
        textColor=colors.HexColor("#003366"),
        spaceAfter=12
    )
    heading_style = ParagraphStyle(
        'HeadingStyle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=colors.HexColor("#003366"),
        spaceAfter=10
    )
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=11,
        textColor=colors.black,
        leading=14,
        spaceAfter=8
    )

    elements = []

    # Title
    elements.append(Paragraph("AI-Generated Report", title_style))
    elements.append(Spacer(1, 12))

    # Introduction
    elements.append(Paragraph("Introduction", heading_style))
    elements.append(Paragraph(f"This report was generated based on the prompt: '{prompt}'.", body_style))
    elements.append(Spacer(1, 12))

    # Literature Review
    elements.append(Paragraph("Literature Review", heading_style))
    for summary in file_summaries:
        elements.append(Paragraph(summary, body_style))
    elements.append(Spacer(1, 12))

    # Findings
    elements.append(Paragraph("Findings", heading_style))
    for response in combined_responses:
        # Split response into bullet points
        points = response.split("\n")  # Split by newlines if they exist
        bullet_items = []
        for point in points:
            if point.strip():  # Ensure no empty lines
                bullet_items.append(ListItem(Paragraph(point.strip(), body_style), leftIndent=20))
        elements.append(ListFlowable(bullet_items, bulletType='bullet', start='•'))
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
                plt.title("Bar Chart of Numeric Data")
                plt.tight_layout()
                plt.savefig(bar_chart_path)
                plt.close()
                elements.append(Paragraph("Bar Chart", heading_style))
                elements.append(ReportLabImage(bar_chart_path, width=400, height=300))
                elements.append(Spacer(1, 12))

                # Pie Chart
                pie_chart_path = os.path.join(STATIC_FOLDER, 'pie_chart.png')
                numeric_df.sum().plot(kind='pie', autopct='%1.1f%%', figsize=(6, 6))
                plt.title("Pie Chart of Numeric Columns")
                plt.tight_layout()
                plt.savefig(pie_chart_path)
                plt.close()
                elements.append(Paragraph("Pie Chart", heading_style))
                elements.append(ReportLabImage(pie_chart_path, width=300, height=300))
                elements.append(Spacer(1, 12))

                # Heat Map
                heatmap_path = os.path.join(STATIC_FOLDER, 'heatmap.png')
                sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
                plt.title("Heatmap of Correlations")
                plt.tight_layout()
                plt.savefig(heatmap_path)
                plt.close()
                elements.append(Paragraph("Heatmap", heading_style))
                elements.append(ReportLabImage(heatmap_path, width=400, height=300))
                elements.append(Spacer(1, 12))

            except Exception as e:
                elements.append(Paragraph(f"Error generating visualizations: {str(e)}", body_style))

    # Build the document
    doc.build(elements)
    return pdf_path
def add_footer(canvas, doc):
    """Add a footer with page numbers."""
    canvas.saveState()
    canvas.setFont('Helvetica', 10)
    canvas.drawString(0.5 * inch, 0.5 * inch, f"Page {doc.page}")
    canvas.restoreState()

@app.route('/preview', methods=['POST'])
def preview_report():
    if 'files' not in request.files or len(request.files.getlist('files')) == 0:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist('files')
    prompt = request.form.get('prompt', '')
    report_type = request.form.get('report_type', 'general')
    combined_responses = []
    file_summaries = []
    dfs = []

    for file in files:
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            file_extension = filename.rsplit(".", 1)[1].lower()
            file_content, df = process_file(file_path, file_extension)
            if not file_content.strip():
                file_content = f"No readable content found in {filename}."

            # Append file summaries
            file_summaries.append(f"<b>{filename}:</b> {file_content[:500]}...")

            # Collect DataFrame for visualization if applicable
            if df is not None:
                dfs.append(df)

            # Retrieve insights using OpenAI, now including the dataset content
            response = retrieve_relevant_chunks(prompt, report_type, file_content)
            combined_responses.append(f"--- Insights from {filename} ---\n{response}")

    if not combined_responses:
        return jsonify({"error": "No content could be processed."}), 400

    try:
        # Combine all DataFrames for visualization (if applicable)
        combined_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        # Generate the PDF with visualizations
        pdf_path = generate_detailed_pdf_with_visualizations(prompt, combined_responses, file_summaries, combined_df)
        return jsonify({"preview_url": f"/static/{os.path.basename(pdf_path)}"})
    except Exception as e:
        return jsonify({"error": f"Error generating PDF: {str(e)}"}), 500
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_file(os.path.join(STATIC_FOLDER, filename))

@app.route("/")
def index():
    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True, port=8000)