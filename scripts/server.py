import os
import re
import logging
import traceback
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from hdbcli import dbapi
import requests
from Intellibase.finalresponse import combine_function_results, generate_aika_prompt
from Intellibase.llm_client import execute_aika_analysis_Intellibase
from destination_srv import get_destination_service_credentials, generate_token, fetch_destination_details, extract_hana_credentials, extract_aicore_credentials
from xsuaa_srv import get_xsuaa_credentials, require_auth
from logger_setup import get_logger

logger = get_logger()
load_dotenv()

# Define error handling system
class ErrorCategory:
    """Enum-like class to categorize errors for frontend interpretation"""
    INPUT_VALIDATION = "input_validation"
    SECURITY = "security"
    RATE_LIMIT = "rate_limit"
    DATABASE = "database"
    PROCESSING = "processing"
    INTERNAL = "internal"
    METHOD_NOT_ALLOWED = "method_not_allowed"

class AppError(Exception):
    """Enhanced application error with standardized structure"""
    def __init__(self, error_type, message, user_friendly=True, status_code=400, details=None):
        super().__init__(message)
        self.error_type = error_type
        self.user_friendly = user_friendly
        self.status_code = status_code
        self.details = details or {}
    
    def to_dict(self):
        """Convert error to standardized dictionary format"""
        error_dict = {
            "error": True,
            "error_type": self.error_type,
            "message": str(self) if self.user_friendly else "An unexpected error occurred. Please try again later.",
            "status_code": self.status_code
        }
        if self.details and self.user_friendly:
            error_dict["details"] = self.details
        return error_dict

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Detect environment
IS_PRODUCTION = 'VCAP_SERVICES' in os.environ
if IS_PRODUCTION:
    logger.info("Running in production mode")
else:
    logger.info("Running in Local mode")

# Set base path and directories
LOCALPATH = os.getenv('LOCALPATH', os.getcwd())
logger.info(f"Base path set: {LOCALPATH}")
base_path = os.getenv('LOCALPATH', os.path.abspath(os.getcwd()))
logger.info(f"Base path configured: {base_path}")
logs_dir = os.path.join(base_path, "logs")
logger.info(f"Logs directory set: {logs_dir}")

# Ensure directories exist
os.makedirs(logs_dir, exist_ok=True)
logger.info(f"Ensured directory exists: {logs_dir}")

# Configure logging with rotation
log_file_path = os.path.join(logs_dir, "TreasuryAnalysis.log")
logger = logging.getLogger('TreasuryAnalysis')
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(log_file_path, maxBytes=50 * 1024 * 1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info("Logging configured with rotation")

# Rate limiter
limiter = Limiter(app=app, key_func=get_remote_address)
logger.info("Rate limiter initialized")

# ---------------------------- XSUAA Authentication Setup ----------------------------
vcap_services = os.environ.get("VCAP_SERVICES")
uaa_xsuaa_credentials = get_xsuaa_credentials(vcap_services)
logger.info(f"XSUAA credentials loaded: {uaa_xsuaa_credentials}")
app.uaa_xsuaa_credentials = uaa_xsuaa_credentials

# ---------------------------- LOAD CF VCAP_SERVICES Variables -----------------------------
logger.info("Loading HANA and AIC credentials from destination services")
destination_service_credentials = get_destination_service_credentials(vcap_services)
logger.info(f"Destination service credentials loaded: {destination_service_credentials}")

try:
    oauth_token = generate_token(
        uri=destination_service_credentials['dest_auth_url'] + "/oauth/token",
        client_id=destination_service_credentials['clientid'],
        client_secret=destination_service_credentials['clientsecret']
    )
    logger.info("OAuth token generated successfully")
except requests.exceptions.HTTPError as e:
    if e.response is not None and e.response.status_code == 500:
        raise Exception("HTTP 500: Check if the client secret is correct.") from e
    else:
        raise

logger.info(f"OAuth token: {oauth_token}")

# -------------------------------- READ HANA DB Configuration -------------------------------------
dest_HDB = 'GENAI_HDB'
hana_dest_details = fetch_destination_details(
    destination_service_credentials['dest_base_url'],
    name=dest_HDB,
    token=oauth_token
)
logger.info(f"HANA destination details fetched: {hana_dest_details}")

HANA_CONN = None
GV_HANA_CREDENTIALS = None

def initialize_hana_connection():
    """Initialize HANA DB connection using extracted credentials"""
    global HANA_CONN, GV_HANA_CREDENTIALS
    logger.info("Initializing HANA database connection")
    GV_HANA_CREDENTIALS = extract_hana_credentials(hana_dest_details)
    logger.info(f"HANA credentials extracted: {GV_HANA_CREDENTIALS}")
    try:
        HANA_CONN = dbapi.connect(
            address=GV_HANA_CREDENTIALS['address'],
            port=GV_HANA_CREDENTIALS['port'],
            user=GV_HANA_CREDENTIALS['user'],
            password=GV_HANA_CREDENTIALS['password'],
            encrypt=True,
            sslValidateCertificate=False
        )
        logger.info("HANA database connection established successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing HANA connection: {str(e)}")
        return False

initialize_hana_connection()

# -------------------------------- READ AIC Configuration -------------------------------------
GV_AIC_CREDENTIALS = None

def initialize_aic_credentials():
    """Initialize AIC credentials from VCAP_SERVICES"""
    global GV_AIC_CREDENTIALS
    logger.info("Initializing AIC credentials")
    try:
        dest_AIC = "GENAI_AI_CORE"
        aicore_details = fetch_destination_details(
            destination_service_credentials['dest_base_url'],
            dest_AIC,
            oauth_token
        )
        logger.info("AIC destination details fetched successfully")
        GV_AIC_CREDENTIALS = extract_aicore_credentials(aicore_details)
        logger.info(f"AIC credentials extracted: {GV_AIC_CREDENTIALS}")
        return True
    except Exception as e:
        logger.error(f"Error initializing AIC credentials: {str(e)}")
        return False

initialize_aic_credentials()

# Log all incoming requests
@app.before_request
def log_request_info():
    logger.info(f"Incoming request: {request.method} {request.url} from {request.remote_addr}")

# Enhanced error handlers
@app.errorhandler(AppError)
def handle_app_error(error):
    """Global error handler for AppError exceptions"""
    logger.info(f"Handling AppError: [{error.error_type}] {str(error)}")
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        "error": True,
        "error_type": ErrorCategory.INTERNAL,
        "message": "An internal server error occurred. Please try again later.",
        "status_code": 500
    }), 500

@app.errorhandler(429)
def rate_limit_error(error):
    logger.warning(f"Rate limit exceeded: {request.remote_addr}")
    return jsonify({
        "error": True,
        "error_type": ErrorCategory.RATE_LIMIT,
        "message": "Rate limit exceeded. Please slow down your requests.",
        "status_code": 429
    }), 429

@app.errorhandler(405)
def method_not_allowed_error(error):
    logger.warning(f"Method not allowed: {request.method} on {request.path}")
    return jsonify({
        "error": True,
        "error_type": ErrorCategory.METHOD_NOT_ALLOWED,
        "message": f"Method {request.method} not allowed for {request.path}.",
        "status_code": 405
    }), 405

# Enhanced Input validation
def validate_user_input(user_input):
    """Validate user input for length and security"""
    logger.info(f"Validating user input: '{user_input[:50]}...'")
    try:
        user_input = user_input.strip()
        logger.info("User input stripped")
        if len(user_input) < 3:
            raise AppError(
                ErrorCategory.INPUT_VALIDATION, 
                "Your query is too short. Please provide at least 3 characters.", 
                status_code=400
            )
        if len(user_input) > 500:
            raise AppError(
                ErrorCategory.INPUT_VALIDATION, 
                "Your query is too long. Please limit your input to 500 characters.", 
                status_code=400
            )
        
        sql_patterns = re.compile(
            r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|alternatives|UNION|EXEC|EXECUTE|TRUNCATE|CREATE|RENAME|DESCRIBE|GRANT|REVOKE)\b'
            r'|\b(OR|AND)\s+\d+\s*=\s*\d+|\b(--|#|\/\*|\*\/)|\b(WAITFOR\s+DELAY)\b)',
            re.IGNORECASE
        )
        if sql_patterns.search(user_input):
            raise AppError(
                ErrorCategory.SECURITY, 
                "Your query contains potentially harmful content. Please revise your input and avoid SQL-like syntax.", 
                status_code=400
            )
        logger.info("User input validated successfully")
        return True
    except AppError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in input validation: {str(e)}")
        raise AppError(
            ErrorCategory.INTERNAL, 
            "We encountered an issue processing your input.", 
            user_friendly=False, 
            status_code=500
        )

# Prompt processing function for Intellibase
def promptlaunchpad(chatquery, user_id):
    """Handle Intellibase prompt processing with error handling"""
    logger.info(f"Starting Intellibase prompt processing for query: '{chatquery[:50]}...'")
    try:
        validate_user_input(chatquery)
        logger.info("User input validated for Intellibase prompt")
        aika_prompt = generate_aika_prompt(chatquery)
        logger.info("AIKA prompt generated")
        chatquery = execute_aika_analysis_Intellibase(aika_prompt)
        logger.info("AIKA analysis executed")
        logger.info(f"Processing query: {chatquery}")
        response = combine_function_results(user_id, chatquery)
        logger.info("Intellibase prompt processing completed successfully")
        logger.debug(f"Query processed successfully")
        return response
    except AppError as e:
        error_dict = e.to_dict()
        logger.info(f"Intellibase prompt processing failed with AppError: {error_dict['message']}")
        return {
            "analysis_result": f"Error: {error_dict['message']}", 
            "sql_query": "N/A", 
            "error": error_dict
        }
    except Exception as e:
        logger.error(f"Unexpected error in Intellibase prompt processing: {str(e)}")
        error_dict = AppError(
            ErrorCategory.PROCESSING, 
            "We couldn't process your query at this time.", 
            user_friendly=True, 
            status_code=500
        ).to_dict()
        return {
            "analysis_result": f"Error: {error_dict['message']}", 
            "sql_query": "N/A", 
            "error": error_dict
        }

@app.route('/api/chat', methods=['POST'])
@require_auth
def chat():
    """Process Intellibase chat queries and return responses"""
    logger.info("Processing Intellibase chat request")
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            logger.warning("Invalid request data")
            raise AppError(
                ErrorCategory.INPUT_VALIDATION,
                "No message provided in your request",
                status_code=400
            )
        
        raw_message = data.get('message', '')
        user_id = None
        user_input = raw_message
        
        # Check if message contains user_id pattern: "user_id: <value> : <actual_message>"
        user_id_pattern = r'^user_id\s*:\s*([^:]+)\s*:\s*(.*)$'
        match = re.match(user_id_pattern, raw_message.strip())
        if match:
            user_id = match.group(1).strip()
            user_input = match.group(2).strip()
            logger.info(f"Extracted user_id: {user_id}")
            logger.info(f"Extracted message: {user_input}")
        else:
            # Fallback to IP address if no user_id is provided
            user_id = request.remote_addr
            logger.info(f"No user_id found in message, using IP address: {user_id}")
        
        # Validate input
        validate_user_input(user_input)
        
        # Process with Intellibase
        try:
            logger.info("Processing query with Intellibase")
            result = promptlaunchpad(user_input, user_id)
            
            if "error" in result:
                error_data = result["error"]
                logger.info(f"Intellibase processing failed: {error_data['message']}")
                return jsonify({
                    "error": True,
                    "error_type": error_data["error_type"],
                    "message": error_data["message"],
                    "FINAL_RESULT": result["analysis_result"],
                    "SQL_QUERY": result["sql_query"]
                }), error_data["status_code"]
            
            formatted_response = {
                "success": True,
                "FINAL_RESULT": str(result["analysis_result"]),
                "SQL_QUERY": result["sql_query"]
            }
            logger.info("Intellibase response generated successfully")
            return jsonify(formatted_response), 200
            
        except Exception as e:
            logger.error(f"Intellibase processing failed: {str(e)}")
            raise AppError(
                ErrorCategory.PROCESSING,
                "Intellibase processing failed",
                status_code=500,
                details={"error": str(e)}
            )
            
    except AppError:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}\n{traceback.format_exc()}")
        raise AppError(
            ErrorCategory.INTERNAL,
            "An unexpected error occurred while processing your request",
            user_friendly=True,
            status_code=500
        )

@app.route('/api/', methods=['GET'])
def home():
    """Return server status"""
    logger.info("Accessing root status endpoint")
    try:
        logger.info("Server status check completed")
        return jsonify({"status": "running"}), 200
    except Exception as e:
        logger.error(f"Server status check failed: {str(e)}")
        return jsonify({"status": "failed", "message": f"Server error: {str(e)}"}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    logger.info("Accessing health check endpoint")
    return jsonify({
        "status": "Server is running",
        "aic_credentials_loaded": GV_AIC_CREDENTIALS is not None,
        "hana_connected": HANA_CONN is not None
    }), 200

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get detailed status information"""
    logger.info("Accessing status endpoint")
    status = {
        "server_status": "running",
        "aic_configuration": {
            "credentials_loaded": GV_AIC_CREDENTIALS is not None,
            "base_url_configured": GV_AIC_CREDENTIALS.get('aic_base_url') is not None if GV_AIC_CREDENTIALS else False,
            "auth_url_configured": GV_AIC_CREDENTIALS.get('aic_auth_url') is not None if GV_AIC_CREDENTIALS else False
        },
        "hana_configuration": {
            "connected": HANA_CONN is not None,
            "credentials_loaded": GV_HANA_CREDENTIALS is not None,
        }
    }
    logger.info("Status information retrieved")
    return jsonify(status), 200

@app.route('/', methods=['GET'])
def root():
    """Root endpoint - redirect to API status or serve a welcome page"""
    logger.info("Accessing root endpoint")
    try:
        return jsonify({
            "message": "Flask Server is running - Intellibase Only",
            "status": "active",
            "api_base": "/api/",
            "available_endpoints": [
                "/api/ - Server status",
                "/api/health - Health check", 
                "/api/chat - Intellibase chat endpoint (POST)",
                "/api/status - Detailed status"
            ]
        }), 200
    except Exception as e:
        logger.error(f"Root endpoint error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting Flask application on port {port}")
    try:
        for rule in app.url_map.iter_rules():
            methods = rule.methods if rule.methods else set()
            logger.info(f"Registered route: {rule} (Methods: {', '.join(sorted(methods))})")
        app.run(host='0.0.0.0', port=port, debug=not IS_PRODUCTION)
    except Exception as e:
        logger.error(f"Failed to start Flask application: {str(e)}\n{traceback.format_exc()}")
        raise AppError(
            ErrorCategory.INTERNAL,
            "Failed to start the application",
            status_code=500
        )