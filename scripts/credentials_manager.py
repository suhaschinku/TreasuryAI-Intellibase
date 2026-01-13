"""
Credentials Manager - Singleton pattern for managing credentials
Loads credentials once and shares across all modules
"""
import threading
import os
from logger_setup import get_logger
from destination_srv import get_destination_service_credentials, generate_token, fetch_destination_details, extract_hana_credentials, extract_aicore_credentials

logger = get_logger()


class CredentialsManager:
    """Singleton class to manage credentials across the application"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if not self._initialized:
            self._hana_credentials = None
            self._aic_credentials = None
            self._destination_service_credentials = None
            self._oauth_token = None
            self._initialized = False
    
    def initialize(self):
        """Initialize all credentials from VCAP_SERVICES"""
        if self._initialized:
            logger.info("Credentials already initialized, skipping")
            return
        
        with self._lock:
            if self._initialized:
                return
            
            logger.info("Initializing CredentialsManager")
            
            try:
                # Get VCAP_SERVICES
                vcap_services = os.environ.get("VCAP_SERVICES")
                if not vcap_services:
                    logger.warning("VCAP_SERVICES not found in environment")
                    return
                
                # Load destination service credentials
                self._destination_service_credentials = get_destination_service_credentials(vcap_services)
                logger.info("Destination service credentials loaded")
                
                # Generate OAuth token
                self._oauth_token = generate_token(
                    uri=self._destination_service_credentials['dest_auth_url'] + "/oauth/token",
                    client_id=self._destination_service_credentials['clientid'],
                    client_secret=self._destination_service_credentials['clientsecret']
                )
                logger.info("OAuth token generated successfully")
                
                # Load HANA credentials
                dest_HDB = 'GENAI_HDB'
                hana_dest_details = fetch_destination_details(
                    self._destination_service_credentials['dest_base_url'],
                    name=dest_HDB,
                    token=self._oauth_token
                )
                self._hana_credentials = extract_hana_credentials(hana_dest_details)
                logger.info("HANA credentials loaded successfully")
                
                # Load AIC credentials
                dest_AIC = "GENAI_AI_CORE"
                aicore_details = fetch_destination_details(
                    self._destination_service_credentials['dest_base_url'],
                    dest_AIC,
                    self._oauth_token
                )
                self._aic_credentials = extract_aicore_credentials(aicore_details)
                logger.info("AIC credentials loaded successfully")
                
                self._initialized = True
                logger.info("CredentialsManager initialization complete")
                
            except Exception as e:
                logger.error(f"Error initializing CredentialsManager: {str(e)}")
                raise
    
    def get_hana_credentials(self):
        """Get HANA credentials"""
        if not self._initialized:
            self.initialize()
        return self._hana_credentials
    
    def get_aic_credentials(self):
        """Get AI Core credentials"""
        if not self._initialized:
            self.initialize()
        return self._aic_credentials
    
    def get_destination_service_credentials(self):
        """Get destination service credentials"""
        if not self._initialized:
            self.initialize()
        return self._destination_service_credentials
    
    def get_oauth_token(self):
        """Get OAuth token"""
        if not self._initialized:
            self.initialize()
        return self._oauth_token
    
    def is_initialized(self):
        """Check if credentials are initialized"""
        return self._initialized


# Global instance
credentials_manager = CredentialsManager()
