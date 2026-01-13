import os
from dotenv import load_dotenv
import logging
import tiktoken
from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.models.llm import LLM
from gen_ai_hub.orchestration.models.message import UserMessage
from gen_ai_hub.orchestration.models.template import Template, TemplateValue
from gen_ai_hub.orchestration.service import OrchestrationService
from gen_ai_hub.orchestration.models.azure_content_filter import AzureContentFilter
from destination_srv import get_destination_service_credentials, generate_token, fetch_destination_details, extract_aicore_credentials
from gen_ai_hub.proxy import GenAIHubProxyClient
from gen_ai_hub.orchestration.models.azure_content_filter import AzureContentFilter 
from gen_ai_hub.orchestration.models.content_filtering import ContentFiltering, InputFiltering, OutputFiltering



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

# Initialize AIC Credentials
logger = logging.getLogger(__name__)
logger.info("====> llm_client_intellibase.py -> GET AIC CREDENTIALS <====")
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

AIC_CREDENTIALS = None
dest_AIC = "GENAI_AI_CORE"
aicore_details = fetch_destination_details(
    destination_service_credentials['dest_base_url'],
    dest_AIC,
    oauth_token
)
AIC_CREDENTIALS = extract_aicore_credentials(aicore_details)
logger.info(f"AIC Credentials: {AIC_CREDENTIALS}")

# Initialize Orchestration Service

proxy_client = GenAIHubProxyClient(
    base_url=AIC_CREDENTIALS['aic_base_url'],
    auth_url=AIC_CREDENTIALS['aic_auth_url'],
    client_id=AIC_CREDENTIALS['clientid'],
    client_secret=AIC_CREDENTIALS['clientsecret'],
    resource_group=AIC_CREDENTIALS['resource_group']
)
ORCHESTRATION_SERVICE_URL = AIC_CREDENTIALS['ORCHESTRATION_SERVICE_URL']
ORCHESTRATION_SERVICE = OrchestrationService(api_url=ORCHESTRATION_SERVICE_URL, proxy_client=proxy_client)

# Define Azure Content Filter thresholds
#CONTENT_FILTER = AzureContentFilter(hate=0, sexual=0, self_harm=0, violence=0, PromptShield=True)
#azure_filter = AzureContentFilter(hate=0, sexual=0, self_harm=0, violence=0, PromptShield=True)

# Model configuration
MODEL_CONFIG = LLM(
    name="anthropic--claude-3.5-sonnet",
    parameters={
        'temperature': 0.3,
        'max_tokens': 200000,
        'top_p': 0.9
    }
)


def run_orchestration(prompt, error_context="orchestration"):
    """Run orchestration service with content filtering."""
    try:
        if ORCHESTRATION_SERVICE is None:
            raise ValueError("OrchestrationService not initialized")

        
        template = Template(messages=[UserMessage("{{ ?extraction_prompt }}")])
        #filter_module = ContentFiltering(
            #input_filtering=InputFiltering(filters=[CONTENT_FILTER])
            # output_filtering=OutputFiltering(filters=[CONTENT_FILTER])
        #)
        config = OrchestrationConfig(template=template, llm=MODEL_CONFIG)
        # config = OrchestrationConfig(template=template, llm=MODEL_CONFIG)
        # config.input_filter = CONTENT_FILTER
        # config.output_filter = CONTENT_FILTER
        
        logger.info(f"Running {error_context} with prompt: {prompt[:100]}...")
        # logger.info(f"Running {error_context} with prompt: {prompt}...")
        response = ORCHESTRATION_SERVICE.run(
            config=config,
            template_values=[TemplateValue("extraction_prompt", prompt)]
        )
        
        result = response.orchestration_result.choices[0].message.content
        logger.info(f"Completed {error_context} with result: {result[:100]}...")
        # logger.info(f"Completed {error_context} with result: {result}...")

    # --- Token counting ---
        input_tokens = count_tokens(prompt)
        output_tokens = count_tokens(result)
        total_tokens = input_tokens + output_tokens
        logger.info(f"Token usage for {error_context}: input={input_tokens}, output={output_tokens}, total={total_tokens}")

        print_token_summary({
            #"result": result,
            "input_tokens": count_tokens(prompt),
            "output_tokens": count_tokens(result),
            "total_tokens": count_tokens(prompt) + count_tokens(result)
            })
    # --- Token counting ---
        return result
    # except OrchestrationError as e:
    #     raise Exception(e.module_results['input_filtering']['message'])
    except Exception as e:
        logger.error(f"Error in {error_context}: {str(e)}", exc_info=True)
        raise Exception(f"Error in {error_context}: {str(e)}")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text string using tiktoken.
    
    Args:
        text (str): The text to count tokens for.
    
    Returns:
        int: Number of tokens in the text.
    """
    try:
        encoder = tiktoken.get_encoding("cl100k_base")
        tokens = encoder.encode(text)
        return len(tokens)
    except Exception as e:
        logger.error(f"Token counting failed: {str(e)}")
        return 0

def execute_llm_call(prompt: str, call_type: str) -> dict:
    """
    Execute an LLM call using OrchestrationService with token counting.
    
    Args:
        prompt (str): Input prompt for the LLM.
        call_type (str): Type of call (e.g., 'coda', 'actions', 'query', 'final_analysis').
    
    Returns:
        dict: Contains 'request' (prompt and metadata), 'response' (LLM output), 
              'input_tokens', 'output_tokens', and 'total_tokens'.
    """
    try:
        # Count input tokens
        input_tokens = count_tokens(prompt)
        logger.info(f"Input tokens: {input_tokens}")
        
        # Prepare request metadata
        request = {
            "prompt": prompt,
            "call_type": call_type,
            "model": "anthropic--claude-3.5-sonnet",
            "temperature": 0.3,
            "topP": 0.9,
            "input_tokens": input_tokens
        }
        
        # Configure orchestration
        template = Template(messages=[UserMessage("{{ ?extraction_prompt }}")])
        config = OrchestrationConfig(template=template, llm=MODEL_CONFIG)
        #config.input_filter = CONTENT_FILTER
        #config.output_filter = CONTENT_FILTER
        
        logger.debug(f"Executing LLM call: {call_type}")
        response = ORCHESTRATION_SERVICE.run(
            config=config,
            template_values=[TemplateValue("extraction_prompt", prompt)]
        )
        
        response_text = response.orchestration_result.choices[0].message.content
        
        # Count output tokens
        output_tokens = count_tokens(response_text)
        total_tokens = input_tokens + output_tokens
        
        logger.info(f"LLM call ({call_type}) completed")
        logger.info(f"Output tokens: {output_tokens}")
        logger.info(f"Total tokens: {total_tokens}")
        
        # Print token summary
        print_token_summary({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens
        })
        
        return {
            "request": request,
            "response": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens
        }
        
    except Exception as e:
        logger.error(f"LLM call ({call_type}) failed: {str(e)}")
        raise

def print_token_summary(result: dict) -> None:
    """
    Print a formatted summary of token usage.
    
    Args:
        result (dict): Result dictionary containing token counts.
    """
    print("\n" + "="*50)
    print("TOKEN USAGE SUMMARY")
    print("="*50)
    print(f"Input Tokens:  {result.get('input_tokens', 0):,}")
    print(f"Output Tokens: {result.get('output_tokens', 0):,}")
    print(f"Total Tokens:  {result.get('total_tokens', 0):,}")
    print("="*50 + "\n")


def execute_aika_analysis_Intellibase(coda_prompt):
    """Execute aika analysis."""
    return run_orchestration(coda_prompt, error_context="aika analysis")  