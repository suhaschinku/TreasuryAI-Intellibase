from logger_setup import get_logger
import requests
import json
import os
from dotenv import load_dotenv
import threading
from datetime import datetime, timedelta

# Set up logger
logger = get_logger()

# Token cache for OAuth tokens
class TokenCache:
    """Cache for OAuth tokens to avoid regenerating on every request"""
    def __init__(self):
        self._token = None
        self._expiry = None
        self._lock = threading.Lock()
    
    def get_token(self, uri, client_id, client_secret):
        """Get cached token or generate new one if expired"""
        with self._lock:
            # Return cached token if valid
            if self._token and self._expiry and datetime.now() < self._expiry:
                logger.info("Using cached OAuth token")
                return self._token
            
            # Generate new token
            logger.info("Generating new OAuth token")
            response = requests.post(uri, data={'grant_type': 'client_credentials'}, 
                                    auth=(client_id, client_secret))
            response.raise_for_status()
            token_data = response.json()
            
            self._token = token_data['access_token']
            # Cache for 80% of token lifetime (default 3600s = 1 hour)
            expires_in = token_data.get('expires_in', 3600)
            self._expiry = datetime.now() + timedelta(seconds=expires_in * 0.8)
            
            logger.info(f"OAuth token cached until {self._expiry}")
            return self._token

# Global token cache instance
_token_cache = TokenCache()

# Step 1: Load environment variables from CF VCAP_SERVICES
def get_destination_service_credentials(vcap_services):
# Function common to both local and cloud environments
# Function to extract destination service credentials from VCAP_SERVICES
    vcap_services = json.loads(vcap_services)
    if not isinstance(vcap_services, dict):
        raise ValueError("VCAP_SERVICES could not be loaded as a dictionary.")
    # Directly access the 'destination' service
    destination_services = vcap_services.get('destination')
    if destination_services and isinstance(destination_services, list) and len(destination_services) > 0:
        creds = destination_services[0].get('credentials', {})
        if all([creds.get('url'), creds.get('clientid'), creds.get('clientsecret'), creds.get('uri')]):
            return {
                'dest_auth_url': creds['url'],
                'clientid': creds['clientid'],
                'clientsecret': creds['clientsecret'],
                'dest_base_url': creds['uri']
            }
    else:
        logger.info("VCAP_SERVICES not found in environment")
        return None

# Step 2: Generate Token for Destination Services (with caching)
def generate_token(uri, client_id, client_secret):
    """Generate OAuth token using cache to avoid unnecessary requests"""
    return _token_cache.get_token(uri, client_id, client_secret)

# Step 3: Get Hana DataBase Details by passing Service Name
def fetch_destination_details(uri, name, token):
    url = f"{uri}/destination-configuration/v1/destinations/{name}"
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(url, headers=headers)
    # response.raise_for_status()
    return response.json()

# Step 4: Extract HANA connection details
def extract_hana_credentials(config):
    dc = config.get('destinationConfiguration', {})
    url = dc.get('URL', '')
    # Remove http:// or https:// if present
    address = url.replace("https://", "").replace("http://", "")
    return {
        'address': address,
        'user': dc.get('User'),
        'password': dc.get('Password'),
        'port': dc.get('Port'),
        #'schema': dc.get('schema')
        'schema': 'TREASURY_AI'
    }

# Step 5: Extract AI Core connection details
def extract_aicore_credentials(config):
    dc = config.get('destinationConfiguration', {})
    return {
        'aic_base_url': dc.get('URL', 'url'),
        'clientid': dc.get('clientId', 'ClientId'),
        'clientsecret': dc.get('clientSecret', 'ClientSecret'),
        'aic_auth_url': dc.get('tokenServiceURL'),
        'resource_group': dc.get('resourceGroup'),
        'ORCHESTRATION_SERVICE_URL': dc.get('ORCHESTRATION_SERVICE_URL')
   }
# Step 6: Extract CAP Credentials
def extract_cap_credentials(config):
    dc = config.get('destinationConfiguration', {})
    return {
        'cap_base_url': dc.get('URL', 'url'),
        'cap_clientid': dc.get('clientId', 'ClientId'),
        'cap_clientsecret': dc.get('clientSecret', 'ClientSecret'),
        'cap_auth_url': dc.get('tokenServiceURL'),
        'resource_group': dc.get('resourceGroup')
    }
