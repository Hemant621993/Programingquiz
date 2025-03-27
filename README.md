# Programming Quiz Application

A comprehensive assessment tool that supports multiple programming languages (Python, C++, C, Java), integrates with Google Gemini AI for generating programming exercises and evaluating user submissions, and provides webcam-based proctoring using YOLOv5 to detect cheating behaviors.

## Features

### Core Functionality
- **Multi-language Support**: Create quizzes for Python, C++, C, and Java
- **AI-Generated Questions**: Uses Google Gemini to create programming challenges with varying difficulty levels
- **Automatic Evaluation**: AI evaluates user submissions and provides feedback
- **Code Testing**: Users can test their code before final submission
- **In-page Execution**: Run and test code within the browser 
- **Webcam Proctoring**: Detect cheating behaviors using computer vision
- **Dynamic Timer**: Timing based on problem complexity determined by the LLM

### Proctoring Capabilities
- Detects suspicious objects (phones, secondary screens, books, etc.)
- Alerts when the user is not visible or looking away
- Takes screenshot evidence of suspicious activities
- Logs all violations with timestamps

### Quiz Management
- **Difficulty Levels**: Easy, Medium, and Hard questions
- **Language-specific Routes**: Share links for specific language assessments
- **Session Tracking**: Track scores across quiz attempts
- **Custom Time Limits**: Based on question difficulty

## Technical Architecture

### Technologies Used
- **Backend**: Flask (Python)
- **AI Integration**: Google Gemini API
- **Computer Vision**: YOLOv5 + OpenCV for face detection
- **Frontend**: Bootstrap, Ace Editor for code editing
- **Containerization**: Docker and Docker Compose for deployment
- **Web Server**: Gunicorn for production-ready serving

### Key Components
1. **Question Generation**: Using Gemini to create contextual problems
2. **Code Evaluation**: AI-powered code review and feedback
3. **Proctoring System**: Multi-stage person and object detection
4. **Timer System**: Intelligent timing based on problem complexity
5. **Docker Environment**: Containerized deployment with environment isolation and consistent dependencies

## How to Use

### Creating a Quiz
1. Access the application at the root URL
2. Select a programming language (/quiz/python, /quiz/cpp, /quiz/c, or /quiz/java)
3. Optionally specify difficulty level (/quiz/python/easy, /quiz/python/medium, /quiz/python/hard)

### Taking a Quiz
1. Select a programming challenge from the list
2. Write your solution in the code editor
3. Test your code using the "Test Code" button
4. Submit your final solution when ready
5. Complete the quiz before the timer expires

### Proctoring Guidelines
- Ensure your face is visible in the webcam
- Don't use phones, books, or other resources unless permitted
- Stay in frame during the entire assessment
- Violations are logged and reported

## Timer Feature
- **Dynamic Timing**: The LLM assigns appropriate time limits based on question difficulty
- **Visual Indicators**: Timer changes color as time runs low
- **Auto-submission**: Solutions are submitted automatically when time expires
- **Session Persistence**: Timer persists across page refreshes

## Setup Instructions

### Prerequisites
- Python 3.8+ (for local development)
- Docker and Docker Compose (for containerized deployment)
- Google Gemini API key

### Environment Variables
- `GEMINI_API_KEY`: Your Google Gemini API key
- `SESSION_SECRET`: Secret key for Flask session management
- `OPENAI_API_KEY`: (Optional) For OpenAI integration if needed

### Option 1: Running Locally
1. Install required Python packages (see dependencies.md)
2. Set the required environment variables in your environment or .env file
3. Run `python main.py` or use `gunicorn` for production
4. Access the application at `http://localhost:5000`

### Option 2: Running with Docker (Recommended)

#### Method A: Using the Deployment Script
1. Clone the repository
2. Run the deployment script:
   ```
   ./deploy.sh
   ```
3. The script will:
   - Check for Docker and Docker Compose
   - Create a .env file from .env.example if needed
   - Build and start the Docker containers
4. Access the application at `http://localhost:5000`

#### Method B: Manual Docker Setup
1. Clone the repository
2. Copy `.env.example` to `.env` and fill in your API keys:
   ```
   cp .env.example .env
   ```
3. Build and start the Docker container:
   ```
   docker-compose up -d
   ```
4. Access the application at `http://localhost:5000`
5. To stop the application:
   ```
   docker-compose down
   ```

### Docker Environment Setup Details
The application is containerized using Docker with the following features:
- Automatic dependency installation
- Persistent storage for webcam captures
- Environment variable management
- Hot-reloading for development

#### Docker Commands
- Build the image: `docker-compose build`
- Start services: `docker-compose up -d`
- View logs: `docker-compose logs -f`
- Stop services: `docker-compose down`
- Rebuild and restart: `docker-compose up -d --build`

### Creating Custom Quizzes
The application can generate quizzes with different parameters:
- Change language by modifying URL path (/quiz/python, /quiz/java, etc.)
- Specify difficulty level in URL (/quiz/python/easy)
- Custom parameters can be added via query parameters

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.