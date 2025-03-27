# Project Dependencies

## Python Dependencies
The following packages are required for this project:

```
flask
gunicorn
werkzeug
google-generativeai
torch
numpy
opencv-python
opencv-python-headless
flask-sqlalchemy
email-validator
psycopg2-binary
openai
Pillow
protobuf
pydantic
requests
```

## How to Install Dependencies in Replit

In Replit, you should use the package manager to install dependencies instead of directly editing requirements.txt. You can do this by:

1. Going to the "Packages" tab in the left sidebar
2. Searching for each package and clicking "+" to install it
3. Replit will automatically add the package to the environment

## Environment Variables

The project requires the following environment variables:

- `GEMINI_API_KEY`: Your Google Gemini API key
- `OPENAI_API_KEY`: (Optional) Only if using OpenAI features
- `SESSION_SECRET`: A secret key for the Flask application

## Directory Structure

- `main.py`: The main application file
- `static/`: Static assets (CSS, JS, images)
- `templates/`: HTML templates
- `webcam_captures/`: Directory for webcam captures and evidence
- `yolov5/`: YOLOv5 model files

## Additional Notes

- The YOLOv5 model file (yolov5s.pt) should be placed in the root directory
- The webcam_captures directory is created automatically if it doesn't exist
- The application runs on port 5000 by default