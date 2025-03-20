from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
import tempfile
import PyPDF2
import logging
from typing import Optional, List, Dict, Any

# Initialize FastAPI app
app = FastAPI(title="Resume and Job Description Parser API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set Groq API key
GROQ_API_KEY ="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"
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler and a stream handler
file_handler = logging.FileHandler('app.log')
stream_handler = logging.StreamHandler()

# Create a formatter and set it for the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

class ResumeInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    technical_skills: List[str] = []
    soft_skills: List[str] = []
    experience: Dict[str, Any] = {}
    education: Dict[str, Any] = {}
    projects: List[Dict[str, Any]] = []

class JobInfo(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    required_skills: List[str] = []
    preferred_skills: List[str] = []
    experience_requirements: Dict[str, Any] = {}
    education_requirements: Dict[str, Any] = {}
    responsibilities: List[str] = []

async def extract_text_from_pdf(pdf_file: UploadFile) -> str:
    """Extract text content from uploaded PDF file"""
    try:
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            # Write the uploaded file content to the temporary file
            temp_file.write(await pdf_file.read())
            temp_path = temp_file.name
        
        logger.info(f"Saved uploaded PDF to temporary file: {temp_path}")

        # Extract text from the PDF
        text = ""
        with open(temp_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        logger.info(f"Extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting text from PDF: {str(e)}")

async def query_groq(text: str, is_resume: bool) -> Dict[str, Any]:
    """Query Groq API to extract information from text"""
    
    if is_resume:
        system_prompt = """
        You are an expert resume parser. Extract the following information from the resume text:
        1. Name of the candidate
        2. Email address
        3. Phone number
        4. Technical skills (list)
        5. Soft skills (list)
        6. Experience details:
           - Total years of experience
           - Job roles (list with company, title, duration)
           - Industry background
        7. Education details:
           - Degree(s)
           - University/Institution(s)
           - Tier ranking (if identifiable)
        8. Projects:
           - Name
           - Technologies used
           - Brief description
        
        Return the data in a structured JSON format with these exact keys:
        {
            "name": "string",
            "email": "string",
            "phone": "string",
            "technical_skills": ["skill1", "skill2", ...],
            "soft_skills": ["skill1", "skill2", ...],
            "experience": {
                "total_years": number,
                "job_roles": [
                    {"company": "string", "title": "string", "duration": "string"}
                ],
                "industry_background": "string"
            },
            "education": {
                "degrees": ["string"],
                "institutions": ["string"],
                "tier_ranking": "string"
            },
            "projects": [
                {
                    "name": "string",
                    "technologies": ["string"],
                    "description": "string"
                }
            ]
        }
        """
        logger.info("Processing resume text")
    else:
        system_prompt = """
        You are an expert job description analyzer. Extract the following information from the job description text:
        1. Job title
        2. Company name
        3. Required skills (list)
        4. Preferred/nice-to-have skills (list)
        5. Experience requirements:
           - Minimum years required
           - Specific domain experience needed
        6. Education requirements:
           - Minimum degree required
           - Preferred fields of study
        7. Key responsibilities (list)
        
        Return the data in a structured JSON format with these exact keys:
        {
            "title": "string",
            "company": "string",
            "required_skills": ["skill1", "skill2", ...],
            "preferred_skills": ["skill1", "skill2", ...],
            "experience_requirements": {
                "minimum_years": number,
                "domain_experience": ["string"]
            },
            "education_requirements": {
                "minimum_degree": "string",
                "preferred_fields": ["string"]
            },
            "responsibilities": ["string"]
        }
        """
        logger.info("Processing job description text")
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama3-70b-8192",  # Using Llama 3 70B model, adjust as needed
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"}
    }
    
    logger.info(f"Sending request to Groq API with model: {payload['model']}")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(GROQ_API_URL, json=payload, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"Groq API Error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Groq API Error: {response.text}"
                )
            
            result = response.json()
            extracted_info = result["choices"][0]["message"]["content"]
            
            logger.info("Successfully received response from Groq API")
            
            # The response should already be JSON but let's ensure it's parsed
            import json
            try:
                parsed_info = json.loads(extracted_info)
                logger.info("Successfully parsed JSON response")
                return parsed_info
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing error: {str(e)}")
                logger.info(f"Returning unparsed Groq API response")
                return extracted_info  # Return as is if already parsed
    
    except httpx.RequestError as e:
        logger.error(f"Error communicating with Groq API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error communicating with Groq API: {str(e)}")

@app.post("/parse/resume", response_model=ResumeInfo)
async def parse_resume(resume_file: UploadFile = File(...)):
    """
    Parse a resume PDF and extract key information using Groq API
    """
    logger.info(f"Received resume parsing request: {resume_file.filename}")
    
    if resume_file.content_type != "application/pdf":
        logger.error(f"Invalid file type: {resume_file.content_type}")
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Extract text from PDF
    resume_text = await extract_text_from_pdf(resume_file)
    
    # Process with Groq API
    parsed_info = await query_groq(resume_text, is_resume=True)
    
    logger.info(f"Successfully parsed resume")
    return parsed_info

@app.post("/parse/job-description", response_model=JobInfo)
async def parse_job_description(job_description: str = Form(...)):
    """
    Parse a job description text and extract key information using Groq API
    """
    logger.info(f"Received job description parsing request")
    
    # Process with Groq API
    parsed_info = await query_groq(job_description, is_resume=False)
    
    logger.info(f"Successfully parsed job description")
    return parsed_info

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check request received")
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI application")
    uvicorn.run(app, host="0.0.0.0", port=8000)