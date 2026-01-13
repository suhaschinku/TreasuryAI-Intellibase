from hdbcli import dbapi
from langchain_community.vectorstores import HanaDB
from gen_ai_hub.proxy.langchain.init_models import init_embedding_model
from logger_setup import get_logger
import os
from dotenv import load_dotenv
from env_config import TABLE_NAMES, SCHEMA_NAME
from contextlib import contextmanager
import threading
import atexit
import time
from datetime import datetime, timedelta
import schedule
from destination_srv import get_destination_service_credentials, generate_token, fetch_destination_details, extract_hana_credentials, extract_aicore_credentials
import json
import base64
import typing

load_dotenv()
logger = get_logger()

# Configuration
SCHEMA_NAME = SCHEMA_NAME
base_path = os.getenv('LOCALPATH', '')
DOCUMENTS_FOLDER_PATH = os.path.join(base_path, "Documents")

# --- HANA CREDENTIALS FROM DESTINATION SERVICES ---
vcap_services = os.environ.get("VCAP_SERVICES")
logger.info("===>DB_Connections => GET HANA CREDENTIALS FROM DESTINATION SERVICES<===")

# Extract destination service credentials from VCAP_SERVICES
destination_service_credentials = get_destination_service_credentials(vcap_services)
logger.info(f"Destination Service Credentials: {destination_service_credentials}")

# Generate OAuth token for destination service
try:
    oauth_token = generate_token(
        uri=destination_service_credentials['dest_auth_url'] + "/oauth/token",
        client_id=destination_service_credentials['clientid'],
        client_secret=destination_service_credentials['clientsecret']
    )
    logger.info("OAuth token generated successfully for destination service.")
except Exception as e:
    logger.error(f"Error generating OAuth token: {str(e)}")
    oauth_token = None

# Get the destination details for the HANA DB
HANA_CREDENTIALS = None
AIC_CREDENTIALS = None
if oauth_token:
    dest_HDB = 'GENAI_HDB'  # Destination name for HANA DB
    hana_dest_details = fetch_destination_details(
        uri=destination_service_credentials['dest_base_url'],
        name=dest_HDB,
        token=oauth_token
    )
    logger.info(f"HANA Destination Details: {hana_dest_details}")
    HANA_CREDENTIALS = extract_hana_credentials(hana_dest_details)
    logger.info(f"HANA_CREDENTIALS: {HANA_CREDENTIALS}")
    AIC_CREDENTIALS = extract_aicore_credentials(hana_dest_details)
    logger.info(f"AIC_CREDENTIALS: {AIC_CREDENTIALS}")
else:
    logger.warning("OAuth token not available; HANA credentials not initialized.")

# Custom Connection Pool Implementation
class ConnectionPool:
    def __init__(self, max_connections=20):
        self.max_connections = max_connections
        self.pool = []
        self.lock = threading.Lock()
        self.cleanup_scheduler_started = False

    def _is_connection_alive(self, conn):
        """Check if a database connection is still alive"""
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM DUMMY")
            cursor.close()
            return True
        except Exception as e:
            logger.warning(f"Connection health check failed: {str(e)}")
            return False
    
    def get_connection(self):
        """Fetch a connection from the pool or create a new one if necessary."""
        with self.lock:
            # Check existing connections in pool for health
            while self.pool:
                conn = self.pool.pop()
                if self._is_connection_alive(conn):
                    logger.debug("Reusing healthy connection from pool")
                    return conn
                else:
                    logger.warning("Discarding dead connection from pool")
                    try:
                        conn.close()
                    except:
                        pass
            
            # No healthy connection available, create new one
            logger.debug("Creating a new connection")
            return self._create_connection()

    def release_connection(self, conn):
        """Release a connection back to the pool."""
        with self.lock:
            if len(self.pool) < self.max_connections:
                self.pool.append(conn)
                logger.debug("Connection released back to pool")
            else:
                conn.close()
                logger.debug("Connection closed as pool is full")

    def _create_connection(self):
        """Create a new database connection."""
        try:
            if not HANA_CREDENTIALS or not all([HANA_CREDENTIALS.get(k) for k in ['address', 'user', 'password', 'port']]):
                logger.error("HANA credentials not properly initialized")
                raise Exception("HANA credentials not available")
            conn = dbapi.connect(
                address=HANA_CREDENTIALS['address'],
                port=int(HANA_CREDENTIALS['port']),
                user=HANA_CREDENTIALS['user'],
                password=HANA_CREDENTIALS['password'],
                encrypt=True,
                sslValidateCertificate=False
            )
            # Set schema if provided
            if HANA_CREDENTIALS.get('schema'):
                cursor = conn.cursor()
                cursor.execute(f"SET SCHEMA {HANA_CREDENTIALS['schema']}")
                cursor.close()
            logger.info("Database connection established successfully")
            return conn
        except Exception as e:
            logger.error(f"Failed to establish database connection: {e}")
            raise

    def close_all_connections(self):
        """Close all connections in the pool."""
        with self.lock:
            for conn in self.pool:
                try:
                    conn.close()
                    logger.info("Closed connection from pool")
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
            self.pool.clear()

# Initialize the global connection pool
connection_pool = ConnectionPool(max_connections=20)

def get_db_connection():
    """Fetch a connection from the global connection pool."""
    logger.debug("Fetching connection from pool")
    return connection_pool.get_connection()

def release_db_connection(conn):
    """Release a connection back to the global connection pool."""
    logger.debug("Releasing connection back to pool")
    connection_pool.release_connection(conn)

def close_all_db_connections():
    """Close all connections in the global connection pool."""
    logger.info("Closing all database connections in the pool")
    connection_pool.close_all_connections()
