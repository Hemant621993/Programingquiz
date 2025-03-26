import sys
import os
import base64
import torch
import argparse
import google.generativeai as genai
import logging
import json
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import cv2
import numpy as np
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add YOLOv5 to system path
sys.path.append("./yolov5")

# Import YOLOv5 utilities
try:
    from models.experimental import attempt_load
    from utils.general import non_max_suppression, scale_boxes
    from utils.torch_utils import select_device
    
    # Load YOLOv5 Model
    device = select_device("")  # Selects GPU if available, else CPU
    yolo_model = attempt_load("yolov5s.pt", device)
    yolo_loaded = True
except Exception as e:
    logger.error(f"Error loading YOLOv5: {str(e)}")
    yolo_loaded = False

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "your_secret_key")  # For session tracking

# Configure Google Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Command line arguments parser - used when running directly, not through gunicorn
def parse_arguments():
    # Check if running through gunicorn
    if 'gunicorn' in os.environ.get('SERVER_SOFTWARE', ''):
        # Default to Python when running with gunicorn
        return type('Args', (), {'technology': 'Python'})
    
    parser = argparse.ArgumentParser(description="Programming Quiz Application")
    parser.add_argument('technology', type=str, nargs='?', default='Python',
                        help='Technology to be evaluated (e.g., Python, C++, Java, C)')
    try:
        args = parser.parse_args()
        return args
    except:
        # If argument parsing fails, default to Python
        return type('Args', (), {'technology': 'Python'})

# Cache for generated questions for different technologies
generated_questions = {}

def generate_questions_for_technology(technology, num_questions=3):
    """Generate programming exercises using Google Gemini"""
    if technology in generated_questions:
        return generated_questions[technology]
    
    logger.info(f"Generating questions for {technology}")
    
    difficulty_levels = ["easy", "medium", "hard"]
    questions = []
    
    for i, difficulty in enumerate(difficulty_levels):
        try:
            # Prompt for generating programming questions with structured format instruction
            prompt = f"""Create a programming problem for {technology} with {difficulty} difficulty level.
            
            You must return your response as valid JSON with the following structure:
            {{
                "id": {i+1},
                "title": "Brief title of the problem",
                "description": "Detailed description of the problem",
                "starter_code": "Code template for the user to start with in {technology}",
                "evaluation_criteria": "Specific criteria to evaluate the solution"
            }}
            
            Follow these rules strictly:
            1. Make sure the problem is educational, well-defined, and appropriate for the difficulty level
            2. For starter_code, include basic structure but don't solve the problem
            3. DO NOT include triple backticks (```) or any markdown formatting in your response
            4. Escape all double quotes inside string values
            5. Return ONLY the JSON object, nothing else
            
            IMPORTANT: Your entire response must be a valid JSON object that can be parsed directly."""
            
            # Set Gemini parameters for structured response
            response = gemini_model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.2,  # Lower temperature for more deterministic response
                    "top_p": 0.8,
                    "top_k": 40,
                    "response_mime_type": "application/json",  # Request JSON response
                }
            )
            
            try:
                # First try direct JSON parsing
                question = json.loads(response.text)
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from the response
                question_data = response.text
                logger.debug(f"Raw question data: {question_data[:100]}...")
                
                # Clean up the response trying different extraction methods
                if "```json" in question_data:
                    question_data = question_data.split("```json")[1].split("```")[0].strip()
                elif "```" in question_data:
                    question_data = question_data.split("```")[1].strip()
                
                # Find where the JSON object might start
                start_idx = question_data.find("{")
                end_idx = question_data.rfind("}")
                
                if start_idx >= 0 and end_idx >= 0:
                    question_data = question_data[start_idx:end_idx+1]
                    
                # Now try to parse the JSON
                question = json.loads(question_data)
                
            questions.append(question)
            
        except Exception as e:
            logger.error(f"Error generating question: {str(e)}")
            # Fallback question if generation fails
            questions.append({
                "id": i+1,
                "title": f"{technology} Exercise {i+1}",
                "description": f"Write a simple program in {technology}.",
                "starter_code": "// Your code here" if technology in ["C", "C++", "Java"] else "# Your code here",
                "evaluation_criteria": "Code correctness and efficiency."
            })
    
    generated_questions[technology] = questions
    return questions

def evaluate_code_with_gemini(code, question, language):
    """Evaluate code submission using Google Gemini"""
    try:
        prompt = f"""
        Evaluate the following {language} code solution for this programming problem:
        
        Problem: {question['title']}
        Description: {question['description']}
        Evaluation Criteria: {question['evaluation_criteria']}
        
        Code Submission:
        ```
        {code}
        ```
        
        You must return your evaluation in valid JSON format with the following structure:
        {{
            "correct": true/false,
            "score": number between 0-10,
            "feedback": "detailed feedback explaining strengths and weaknesses",
            "suggestions": "improvement suggestions"
        }}
        
        Follow these rules strictly:
        1. Provide thoughtful, educational feedback appropriate for learning
        2. DO NOT include triple backticks (```) or any markdown formatting in your response
        3. Escape all double quotes inside string values
        4. Return ONLY the JSON object, nothing else
        5. Base your score on correctness, efficiency, readability, and best practices
        
        IMPORTANT: Your entire response must be a valid JSON object that can be parsed directly.
        """
        
        # Set Gemini parameters for structured response
        response = gemini_model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,  # Lower temperature for more deterministic response
                "top_p": 0.8,
                "top_k": 40,
                "response_mime_type": "application/json",  # Request JSON response
            }
        )
        
        try:
            # First try direct JSON parsing
            evaluation = json.loads(response.text)
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON from the response
            result_text = response.text
            logger.debug(f"Raw evaluation data: {result_text[:100]}...")
            
            # Clean up the response trying different extraction methods
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].strip()
            
            # Find where the JSON object might start and end
            start_idx = result_text.find("{")
            end_idx = result_text.rfind("}")
            
            if start_idx >= 0 and end_idx >= 0:
                result_text = result_text[start_idx:end_idx+1]
                
            # Now try to parse the JSON
            evaluation = json.loads(result_text)
        
        # Ensure all required fields are present
        required_fields = ["correct", "score", "feedback", "suggestions"]
        for field in required_fields:
            if field not in evaluation:
                if field == "correct":
                    evaluation[field] = False
                elif field == "score":
                    evaluation[field] = 0
                else:
                    evaluation[field] = f"No {field} provided"
        
        return evaluation
        
    except Exception as e:
        logger.error(f"Error evaluating code: {str(e)}")
        return {
            "correct": False,
            "score": 0,
            "feedback": f"Evaluation error: {str(e)}",
            "suggestions": "Please try again."
        }

# Main route redirects to Python by default
@app.route("/")
def index():
    return redirect(url_for("quiz_by_language", language="python"))

# Routes for specific programming languages
@app.route("/quiz/<language>")
def quiz_by_language(language):
    # Convert URL parameter to proper format
    language_mapping = {
        "python": "Python",
        "cpp": "C++",
        "c": "C",
        "java": "Java"
    }
    technology = language_mapping.get(language.lower(), "Python")
    
    # Get questions for the selected technology
    questions = generate_questions_for_technology(technology)
    return render_template("question_selector.html", questions=questions, technology=technology)

@app.route("/quiz/<language>/question/<int:question_id>")
def show_editor(language, question_id):
    language_mapping = {
        "python": "Python",
        "cpp": "C++",
        "c": "C",
        "java": "Java"
    }
    technology = language_mapping.get(language.lower(), "Python")
    
    questions = generate_questions_for_technology(technology)
    question = next((q for q in questions if q["id"] == question_id), None)
    if question is None:
        return "Question not found", 404
    return render_template("code_editor.html", question=question, technology=technology, language=language)

@app.route("/quiz/<language>/submit_code", methods=["POST"])
def submit_code(language):
    try:
        language_mapping = {
            "python": "Python",
            "cpp": "C++",
            "c": "C",
            "java": "Java"
        }
        technology = language_mapping.get(language.lower(), "Python")
        
        data = request.json
        source_code = data.get("code", "")
        question_id = int(data.get("question_id"))
        
        questions = generate_questions_for_technology(technology)
        question = next((q for q in questions if q["id"] == question_id), None)
        if not question:
            return jsonify({"error": "Invalid question ID"}), 400

        # Evaluate code using Gemini
        evaluation = evaluate_code_with_gemini(source_code, question, technology)
        
        # Store score in session with language key to track per-language scores
        session_key = f"scores_{language}"
        if session_key not in session:
            session[session_key] = {}
        scores = session[session_key]
        scores[str(question_id)] = evaluation["score"]
        session[session_key] = scores
        
        response_data = {
            "output": f"Evaluation complete",
            "status": "Accepted" if evaluation["correct"] else "Failed",
            "score": evaluation["score"],
            "feedback": evaluation["feedback"],
            "suggestions": evaluation["suggestions"],
            "correct": evaluation["correct"]
        }
        
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error in submit_code: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route("/quiz/<language>/result")
def show_result(language):
    language_mapping = {
        "python": "Python",
        "cpp": "C++",
        "c": "C",
        "java": "Java"
    }
    technology = language_mapping.get(language.lower(), "Python")
    
    session_key = f"scores_{language}"
    scores = session.get(session_key, {})
    questions = generate_questions_for_technology(technology)
    total_score = sum(scores.values()) if scores else 0
    max_score = len(questions) * 10
    
    return render_template("submitted.html", 
                          total_score=total_score, 
                          max_score=max_score, 
                          scores=scores, 
                          questions=questions,
                          technology=technology,
                          language=language)

@app.route("/upload_webcam", methods=["POST"])
def upload_webcam():
    if not yolo_loaded:
        return jsonify({"status": "success", "message": "YOLOv5 not loaded, webcam monitoring disabled", "suspicious": False})
    
    try:
        data = request.json
        image_data = data["image"].split(",")[1]
        image_bytes = base64.b64decode(image_data)

        if not os.path.exists("webcam_captures"):
            os.makedirs("webcam_captures")

        image_path = f"webcam_captures/capture_{len(os.listdir('webcam_captures'))}.png"
        with open(image_path, "wb") as f:
            f.write(image_bytes)

        result = detect_suspicious_activity(image_path)
        return jsonify({"status": "success", "message": "Webcam image saved!", "suspicious": result})
    except Exception as e:
        logger.error(f"Error processing webcam image: {str(e)}")
        return jsonify({"status": "error", "message": f"Error: {str(e)}"})

def detect_suspicious_activity(image_path):
    """Use YOLOv5 to detect suspicious activity in webcam captures"""
    if not yolo_loaded:
        return False
        
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float().to(device)
        img = img.permute(2, 0, 1)
        img = img.unsqueeze(0)
        img /= 255.0

        with torch.no_grad():
            pred = yolo_model(img)[0]

        pred = non_max_suppression(pred, 0.4, 0.5)
        suspicious_classes = {67, 73, 77}
        person_count = 0

        for det in pred:
            if det is not None and len(det):
                for *xyxy, conf, cls in det:
                    class_id = int(cls.item())  
                    if class_id == 0:
                        person_count += 1
                    if class_id in suspicious_classes:
                        return True

        if person_count == 0 or person_count > 1:
            return True

        return False
    except Exception as e:
        logger.error(f"Error in suspicious activity detection: {str(e)}")
        return False

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
