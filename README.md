# Indian_Legal_Advisor
Advices for your legal issues

# README.md
# Indian Legal Assistant

An AI-powered legal assistance tool that helps users understand Indian laws by scraping legal resources and providing analysis.

## Features

- **Legal Web Scraping**: Automatically searches and extracts information from Indian legal websites like Indian Kanoon, Legislative.gov.in, and Supreme Court.
- **Document Processing**: Extracts text from PDFs, images, and text files for legal analysis.
- **AI-Powered Analysis**: Uses Google Gemini AI to provide plain-language explanations of legal concepts.
- **Case Law Retrieval**: Finds relevant case laws and legal sections that apply to the user's issue.
- **User-Friendly Interface**: Clean, responsive design with clear presentation of legal information.

## Setup Instructions

### Prerequisites

- Python 3.9+
- Redis (optional, for caching)
- Tesseract OCR (for image processing)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/indian-legal-assistant.git
cd indian-legal-assistant
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR:
   - **Ubuntu**: `sudo apt install tesseract-ocr`
   - **Windows**: Download and install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`

5. Copy the example environment file and update it with your settings:
```bash
cp .env.example .env
```
   - Get a Google Gemini API key from [Google AI Studio](https://ai.google.dev/)
   - Set a secure SECRET_KEY for Flask

6. Create upload directory:
```bash
mkdir -p static/uploads
```

### Running the Application

```bash
python app.py
```

Visit `http://localhost:5000` in your browser to use the application.

## Ethical and Legal Considerations

- This tool is for **educational purposes only**
- Ensure compliance with Indian IT Act regulations
- Adhere to website terms of service during web scraping 
- Use rate limiting to prevent server overload
- Proper attribution of legal sources
- Always include disclaimers that this is not substitute for professional legal advice

## Project Structure

```
indian-legal-assistant/
├── app.py               # Main Flask application
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (not tracked in git)
├── .gitignore           # Git ignore file
├── static/              # Static assets
│   ├── css/
│   │   └── style.css    # CSS styles
│   ├── js/
│   │   └── script.js    # JavaScript code
│   └── uploads/         # User uploaded files
├── templates/           # HTML templates
│   ├── index.html       # Homepage
│   ├── result.html      # Results page
│   ├── 404.html         # Error page
│   └── 500.html         # Server error page
└── README.md            # Project documentation
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# sample.env
# Rename this file to .env and fill in your actual keys/settings

# API Keys
GEMINI_API_KEY=your_api_key_here

# Flask Configuration
FLASK_APP=app.py
FLASK_ENV=development
FLASK_DEBUG=False
SECRET_KEY=your_secure_random_string_here

# Redis Cache (optional)
REDIS_HOST=localhost
REDIS_PORT=6379

# Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create upload directory
RUN mkdir -p static/uploads

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5000

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
