import sys
import os
import base64
import torch
import argparse
import google.generativeai as genai
import logging
import json
import subprocess
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
    
    # Create webcam_captures directory for storing evidence
    if not os.path.exists("webcam_captures"):
        os.makedirs("webcam_captures")
        logger.info("Created webcam_captures directory for storing evidence")
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

def generate_questions_for_technology(technology, num_questions=3, difficulty=None):
    """Generate programming exercises using Google Gemini
    
    Args:
        technology (str): Programming language (Python, C++, Java, C)
        num_questions (int): Number of questions to generate
        difficulty (str, optional): Specific difficulty level (easy, medium, hard)
                                    If provided, all questions will be of this difficulty
    """
    # Create a cache key that includes both technology and difficulty
    cache_key = f"{technology}_{difficulty}" if difficulty else technology
    
    # Check cache first
    if cache_key in generated_questions:
        return generated_questions[cache_key]
    
    logger.info(f"Generating questions for {technology}" + (f" with {difficulty} difficulty" if difficulty else ""))
    
    # Define difficulty levels based on parameter
    if difficulty and difficulty.lower() in ["easy", "medium", "hard"]:
        if difficulty.lower() == "easy":
            # For easy, generate 2 easy and 1 medium
            difficulty_levels = ["easy", "easy", "medium"]
        elif difficulty.lower() == "medium":
            # For medium, generate 1 easy, 1 medium, 1 hard
            difficulty_levels = ["easy", "medium", "hard"]
        else:  # hard
            # For hard, generate 1 medium and 2 hard
            difficulty_levels = ["medium", "hard", "hard"]
    else:
        # Default: one of each difficulty level
        difficulty_levels = ["easy", "medium", "hard"]
    
    questions = []
    
    for i, question_difficulty in enumerate(difficulty_levels):
        try:
            # Prompt for generating programming questions with structured format instruction
            prompt = f"""Create a programming problem for {technology} with {question_difficulty} difficulty level.
            
            You must return your response as valid JSON with the following structure:
            {{
                "id": {i+1},
                "title": "Brief title of the problem",
                "description": "Detailed description of the problem",
                "starter_code": "Code template for the user to start with in {technology}",
                "evaluation_criteria": "Specific criteria to evaluate the solution",
                "difficulty": "{question_difficulty}",
                "time_limit_minutes": integer representing reasonable time limit in minutes
            }}
            
            Follow these rules strictly:
            1. Make sure the problem is educational, well-defined, and appropriate for the {question_difficulty} difficulty level
            2. For starter_code, include basic structure but don't solve the problem
            3. DO NOT include triple backticks (```) or any markdown formatting in your response
            4. Escape all double quotes inside string values
            5. Return ONLY the JSON object, nothing else
            6. For time_limit_minutes, assign:
               - 10-15 minutes for easy problems
               - 20-30 minutes for medium problems
               - 40-60 minutes for hard problems
            
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
            
            # Ensure question has difficulty field
            if "difficulty" not in question:
                question["difficulty"] = question_difficulty
                
            questions.append(question)
            
        except Exception as e:
            logger.error(f"Error generating question: {str(e)}")
            # Fallback question if generation fails
            # Set default time limit based on difficulty
            default_time_limits = {
                "easy": 15,
                "medium": 25,
                "hard": 45
            }
            time_limit = default_time_limits.get(question_difficulty.lower(), 20)
            
            questions.append({
                "id": i+1,
                "title": f"{technology} Exercise {i+1}",
                "description": f"Write a simple program in {technology}.",
                "starter_code": "// Your code here" if technology in ["C", "C++", "Java"] else "# Your code here",
                "evaluation_criteria": "Code correctness and efficiency.",
                "difficulty": question_difficulty,
                "time_limit_minutes": time_limit
            })
    
    # Store in cache with the correct key
    generated_questions[cache_key] = questions
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
@app.route("/quiz/<language>/<difficulty>")
def quiz_by_language(language, difficulty=None):
    # Convert URL parameter to proper format
    language_mapping = {
        "python": "Python",
        "cpp": "C++",
        "c": "C",
        "java": "Java"
    }
    technology = language_mapping.get(language.lower(), "Python")
    
    # Validate difficulty level
    valid_difficulties = ["easy", "medium", "hard"]
    if difficulty and difficulty.lower() not in valid_difficulties:
        difficulty = None
    
    # Get questions for the selected technology with optional difficulty
    questions = generate_questions_for_technology(technology, difficulty=difficulty)
    return render_template("question_selector.html", questions=questions, technology=technology, difficulty=difficulty)

@app.route("/quiz/<language>/question/<int:question_id>")
def show_editor(language, question_id):
    language_mapping = {
        "python": "Python",
        "cpp": "C++",
        "c": "C",
        "java": "Java"
    }
    technology = language_mapping.get(language.lower(), "Python")
    
    # Get difficulty from query parameter if provided
    difficulty = request.args.get('difficulty')
    
    questions = generate_questions_for_technology(technology, difficulty=difficulty)
    question = next((q for q in questions if q["id"] == question_id), None)
    if question is None:
        return "Question not found", 404
    return render_template("code_editor.html", question=question, technology=technology, language=language)

@app.route("/quiz/<language>/test_code", methods=["POST"])
def test_code(language):
    """Run code and return the output without evaluation"""
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
        stdin = data.get("stdin", "")
        
        # Create a temporary file for running the code
        temp_dir = os.path.join(os.getcwd(), "temp")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        timestamp = time.strftime("%Y%m%d%H%M%S")
        result = {"success": False, "output": "", "error": None}
        
        # Execute the code according to language
        if technology.lower() == "python":
            # For Python, use a temporary file and subprocess
            file_path = os.path.join(temp_dir, f"test_{timestamp}.py")
            with open(file_path, "w") as f:
                f.write(source_code)
            
            # Execute the code in a subprocess with timeout
            try:
                if stdin:
                    proc = subprocess.run(
                        [sys.executable, file_path],
                        input=stdin,
                        text=True,
                        capture_output=True,
                        timeout=5
                    )
                else:
                    proc = subprocess.run(
                        [sys.executable, file_path],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                
                if proc.returncode == 0:
                    result["success"] = True
                    result["output"] = proc.stdout
                else:
                    result["output"] = proc.stderr
                
            except subprocess.TimeoutExpired:
                result["output"] = "Execution timed out (5 seconds limit)"
            except Exception as e:
                result["error"] = str(e)
                result["output"] = f"Error executing code: {str(e)}"
        
        # For other languages, return appropriate message
        else:
            result["output"] = f"Running {technology} code is not supported in the test environment. " \
                              f"Your code will still be evaluated when you click 'Run Code'."
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in test_code: {str(e)}")
        return jsonify({"success": False, "error": str(e), "output": f"Server error: {str(e)}"}), 500

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
        
        # Get difficulty from data if provided
        difficulty = request.args.get('difficulty')
        
        questions = generate_questions_for_technology(technology, difficulty=difficulty)
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
    
    # Get difficulty from query parameter if provided
    difficulty = request.args.get('difficulty')
    
    session_key = f"scores_{language}"
    scores = session.get(session_key, {})
    questions = generate_questions_for_technology(technology, difficulty=difficulty)
    total_score = sum(scores.values()) if scores else 0
    max_score = len(questions) * 10
    
    return render_template("submitted.html", 
                          total_score=total_score, 
                          max_score=max_score, 
                          scores=scores, 
                          questions=questions,
                          technology=technology,
                          language=language,
                          difficulty=difficulty)

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

        # Save the original capture
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        image_path = f"webcam_captures/capture_{timestamp}.png"
        with open(image_path, "wb") as f:
            f.write(image_bytes)

        # Detect suspicious activity
        result = detect_suspicious_activity(image_path)
        
        # Extract information from result
        is_suspicious = result.get('suspicious', False)
        reason = result.get('reason', 'Unknown reason')
        detected_objects = result.get('objects', [])
        person_detected = result.get('person_detected', False)
        person_count = result.get('person_count', 0)
        
        # Prepare response with detailed information
        message = "No suspicious activity detected"
        if is_suspicious:
            message = f"Warning: {reason}"
            if detected_objects:
                message += f" (Detected: {', '.join(detected_objects)})"
        elif person_detected:
            message = f"Person detected properly ({person_count})"
        
        return jsonify({
            "status": "success", 
            "message": message, 
            "suspicious": is_suspicious,
            "reason": reason,
            "detected_objects": detected_objects,
            "person_detected": person_detected,
            "person_count": person_count
        })
    except Exception as e:
        logger.error(f"Error processing webcam image: {str(e)}")
        return jsonify({"status": "error", "message": f"Error: {str(e)}"})

def detect_suspicious_activity(image_path):
    """Use YOLOv5 to detect suspicious activity in webcam captures"""
    if not yolo_loaded:
        return {'suspicious': False, 'reason': 'YOLOv5 not loaded'}
        
    try:
        # Read and prepare image
        img_orig = cv2.imread(image_path)
        if img_orig is None or img_orig.size == 0:
            return {
                'suspicious': False,
                'reason': "Invalid image file or empty frame",
                'objects': []
            }
            
        img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img).float().to(device)
        img_tensor = img_tensor.permute(2, 0, 1)
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor /= 255.0

        # Run YOLOv5 detection with lower confidence threshold (0.25 instead of 0.4)
        with torch.no_grad():
            pred = yolo_model(img_tensor)[0]

        # Process predictions with lower confidence threshold for better detection
        pred = non_max_suppression(pred, 0.25, 0.45)
        
        # Class IDs for suspicious objects:
        # 67: cell phone, 73: laptop/computer, 77: book/notes, 41: cup (potential for hiding notes)
        # 24-26: various devices, 28: TV/monitor
        suspicious_classes = {67, 73, 77, 41, 24, 25, 26, 28}
        detected_objects = []
        person_count = 0
        suspicious_reason = None
        violation_found = False
        person_detected = False

        # Class names for YOLOv5 model
        class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
                      "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
                      "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
                      "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                      "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
                      "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
                      "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                      "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", 
                      "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                      "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

        # Process detection results
        for det in pred:
            if det is not None and len(det):
                for *xyxy, conf, cls in det:
                    class_id = int(cls.item())
                    confidence = float(conf.item())
                    
                    # Get object name
                    object_name = class_names[class_id] if class_id < len(class_names) else f"object-{class_id}"
                    
                    # Mark the detected item on the image
                    x1, y1, x2, y2 = [int(coord) for coord in xyxy]
                    cv2.rectangle(img_orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_orig, f"{object_name} {confidence:.2f}", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Count persons with high confidence
                    if class_id == 0 and confidence > 0.3:  # person class
                        person_count += 1
                        person_detected = True
                    
                    # Check for suspicious objects with higher confidence threshold
                    if class_id in suspicious_classes and confidence > 0.35:
                        detected_objects.append(object_name)
                        
                        # Change rectangle to red for suspicious objects
                        cv2.rectangle(img_orig, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(img_orig, f"{object_name} {confidence:.2f}", 
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        suspicious_reason = f"Detected suspicious object: {object_name}"
                        violation_found = True
                        
                        # Save the evidence image with timestamp
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        evidence_path = f"webcam_captures/evidence_{timestamp}.jpg"
                        cv2.imwrite(evidence_path, img_orig)
                        
                        # Log the violation
                        logger.warning(f"Cheating detected! Found {object_name} in webcam capture. Evidence saved to {evidence_path}")
                        
                        break  # Exit the detection loop once violation is found
        
        # If YOLO didn't detect a person, try with OpenCV's face detection as a fallback
        if person_count == 0:
            # Convert back to grayscale for face detection
            gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
            
            # Load Haar Cascade for face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect faces with relaxed parameters for better detection
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # If face(s) detected with the cascade, update person detection
            if len(faces) > 0:
                person_detected = True
                person_count = len(faces)
                
                # Mark faces on the image
                for (x, y, w, h) in faces:
                    cv2.rectangle(img_orig, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(img_orig, "Face detected", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Check for no face or multiple faces only if no other violations
        if not violation_found:
            if not person_detected:
                suspicious_reason = "No person detected in frame"
                violation_found = True
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                evidence_path = f"webcam_captures/evidence_noface_{timestamp}.jpg"
                cv2.imwrite(evidence_path, img_orig)
                logger.warning(f"Cheating detected! No face in frame. Evidence saved to {evidence_path}")
            
            elif person_count > 1:
                suspicious_reason = f"Multiple people detected ({person_count})"
                violation_found = True
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                evidence_path = f"webcam_captures/evidence_multiface_{timestamp}.jpg"
                cv2.imwrite(evidence_path, img_orig)
                logger.warning(f"Cheating detected! Multiple faces ({person_count}) in frame. Evidence saved to {evidence_path}")
        
        # Save detection image for debugging (can be removed in production)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        debug_path = f"webcam_captures/debug_{timestamp}.jpg"
        cv2.imwrite(debug_path, img_orig)

        # Return results with reason
        return {
            'suspicious': violation_found,
            'reason': suspicious_reason if violation_found else "No suspicious activity detected",
            'objects': detected_objects,
            'person_detected': person_detected,
            'person_count': person_count
        }
    except Exception as e:
        logger.error(f"Error in person detection: {str(e)}")
        # Return non-suspicious result on error to prevent false positives
        return {
            'suspicious': False,
            'reason': f"Detection error: {str(e)}",
            'objects': []
        }

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
