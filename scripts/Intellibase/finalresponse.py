from dotenv import load_dotenv
import os
from datetime import datetime, date
import json
from typing import List, Dict
from db_connection import get_db_connection, release_db_connection
from Intellibase.llm_client import execute_llm_call, get_logger
import concurrent.futures
from destination_srv import get_destination_service_credentials, generate_token, fetch_destination_details, extract_hana_credentials

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

# Initialize HANA Credentials
logger.info("====> finalresponse.py -> GET HANA CREDENTIALS <====")
vcap_services = os.environ.get("VCAP_SERVICES")
destination_service_credentials = get_destination_service_credentials(vcap_services)
logger.info(f"Destination Service Credentials: {destination_service_credentials}")

try:
    oauth_token = generate_token(
        uri=destination_service_credentials['dest_auth_url'] + "/oauth/token",
        client_id=destination_service_credentials['clientid'],
        client_secret=destination_service_credentials['clientsecret']
    )
    logger.info("OAuth token generated successfully for destination service.")
except Exception as e:
    logger.error(f"Error generating OAuth token: {str(e)}")
    raise

HANA_CREDENTIALS = None
dest_HDB = "GENAI_HDB"
hana_dest_details = fetch_destination_details(
    destination_service_credentials['dest_base_url'],
    dest_HDB,
    oauth_token
)
HANA_CREDENTIALS = extract_hana_credentials(hana_dest_details)
logger.info(f"HANA Credentials: {HANA_CREDENTIALS}")

# Utility Functions

def read_template(filepath: str) -> str:
    """Cached template file reading"""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logger.error(f"Failed to read template {filepath}: {str(e)}")
        raise

def json_serializer(obj):
    """Custom JSON serializer for datetime objects"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return str(obj)

def generate_aika_prompt(input_text):
    """Generate a CODA analysis prompt from a template."""
    LOCALPATH = os.getcwd()
    filepath = os.path.join(LOCALPATH, "Intellibase", "prompt_AIKA.txt")
    logger.info("AIKA Path", filepath)
    logger.info(filepath)
    try:
        coda_text = read_template(filepath)
        return f"Analyse the prompt and come up with a proper prompt in a sentence format. STRICTLY PROVIDE ONLY THE UPDATED PROMPT. Mandatoriy APPEND 'Intellibase' keyword to the prompt.  <method>{coda_text}</method><prompt>{input_text}</prompt>"
    except Exception as e:
        raise Exception(f"An error occurred: {str(e)}")

# Prompt Generation Functions
def codapromptgenerator(input_text: str) -> str:
    LOCALPATH = os.getcwd()
    logger.info(LOCALPATH)
    paths = {
        'coda': os.path.join(LOCALPATH,"Intellibase", "prompt_CODA_INTEL.txt"),
        'datamodel': os.path.join(LOCALPATH, "Intellibase", "datamodel.txt"),
        'datadictionary': os.path.join(LOCALPATH, "Intellibase", "Data_Dictionary.txt")
    }
    
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_text = {executor.submit(read_template, path): key for key, path in paths.items()}
            texts = {key: future.result() for future, key in future_to_text.items()}
        
        combined_text = (
            """STRICTLY Analyse the prompt using the method, the data model supplied and Dictionary model
            and come up with clear steps to analyse, filters to be applied and data needed to perform the analysis."""
            f"<method>{texts['coda']}</method>"
            f"<data model>{texts['datamodel']}</data model>"
            f"<Dictionary model>{texts['datadictionary']}</Dictionary model>"
            f"<prompt>{input_text}</prompt>"
        )
        logger.debug("CODA prompt generated successfully")
        return combined_text
    except Exception as e:
        logger.error(f"CODA prompt generation failed: {str(e)}")
        raise

def askdatapromptgenerator(user_id,input_text: str) -> str:
    LOCALPATH = os.getcwd()
    filepath = os.path.join(LOCALPATH, "Intellibase", "prompt_querygenerator.txt")
    logger.info(filepath)
    logger.info("Prompt Query Generator Path", filepath)
    try:
        existing_text = read_template(filepath)
        existing_text = existing_text.replace('{user_id}',user_id)
        combined_text = f"{existing_text}{input_text}. Provide only the query without any explanations and information."
        logger.debug(f"Generated query prompt for: {input_text}")
        return combined_text
    except Exception as e:
        logger.error(f"Query prompt generation failed: {str(e)}")
        raise

def finalanalysispromptgenerator(actionsteps: str, data: str) -> str:
    LOCALPATH = os.getcwd()
    filepath = os.path.join(LOCALPATH, "Intellibase", "prompt_finalanalysis.txt")
    logger.info("Prompt Final Analysis Path", filepath)
    try:
        prompttext = read_template(filepath)
        return f"<prompt>{prompttext}</prompt><analysis steps>{actionsteps}</analysis steps><data>{data}</data>"
    except Exception as e:
        logger.error(f"Final analysis prompt generation failed: {str(e)}")
        raise

# Data Processing Functions
def dataask(codaresult: str) -> str:
    actionsprompt = f"<prompt> extract the data requirements portion and filters to be applied of the following text </prompt> <text> {codaresult} </text>"
    return execute_llm_call(actionsprompt, "data_ask")["response"]

def actionsteps(codaresult: str) -> str:
    actionsprompt = f"<prompt> extract the required analysis portion of the following text </prompt> <text> {codaresult} </text>"
    return execute_llm_call(actionsprompt, "actions")["response"]

# Query Execution
def validate_query(query: str) -> str:
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE']
    if any(keyword in query.upper() for keyword in dangerous_keywords):
        raise ValueError("Potentially dangerous SQL operation detected")
    return query.strip()

def query_to_json(query: str) -> str:
    validated_query = validate_query(query)
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(validated_query)
            logger.info(f"Executed Query: {validated_query}")
            if cursor.description:
                columns = [description[0] for description in cursor.description]
                results = [dict(zip(columns, row)) for row in cursor.fetchall()]
                json_result = json.dumps(results, default=json_serializer, indent=2)
                logger.info(f"Query returned {len(results)} rows")
                return json_result
            else:
                logger.info("Query executed, but returned no results")
                return json.dumps([])
    except ValueError as val_error:
        logger.error(f"Query validation error: {val_error}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during query execution: {str(e)}")
        raise
    finally:
        if conn:
            release_db_connection(conn)

# Analysis Pipeline
def combine_function_results(user_id,input_text: str) -> dict:
    start_time = datetime.now()
    logger.info(f"Starting analysis pipeline for: {input_text}")
    try:
        # CODA Analysis
        codaprompt = codapromptgenerator(input_text)
        codaresult = execute_llm_call(codaprompt, "coda")["response"]
        
        # Component Extraction (Parallel)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_actions = executor.submit(actionsteps, codaresult)
            future_dataask = executor.submit(dataask, codaresult)
            actions = future_actions.result()
            dataaskresult = future_dataask.result()
        
        # Query Generation and Execution
        queryprompt = askdatapromptgenerator(user_id,dataaskresult)
        logger.info(queryprompt)
        querytohana = execute_llm_call(queryprompt, "query")["response"]
        datajson = query_to_json(querytohana)
        
        # Final Analysis
        finalanalysisprompt = finalanalysispromptgenerator(actions, datajson)
        finalanalysisresult = execute_llm_call(finalanalysisprompt, "final_analysis")["response"]
        
        # Create combined result object
        result = {
            "analysis_result": finalanalysisresult,
            "sql_query": querytohana
        }
        
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Analysis completed in {execution_time} seconds")
        return result
    except Exception as e:
        logger.error(f"Analysis pipeline failed: {str(e)}")
        raise