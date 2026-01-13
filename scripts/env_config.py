import os
import json
from dotenv import load_dotenv

# Table names for HANA DB
TABLE_NAMES = {
    "transcript": "TREASURY_EMBEDD_TRANSCRIPT_UI5",
    "non_transcript": "TREASURY_EMBEDD_NON_TRANSCRIPT_UI5",
    "excel_non_transcript": "TREASURY_EMBEDD_EXCEL_NON_TRANSCRIPT_UI5"
}

#Hana Treasury Table
TREASURY_TABLE = "TREASURY_AI_INTELLIBASE_VIEW"
SCHEMA_NAME = "TREASURY_AI"

# Embedding model
HANA_DB_API ="3cb8ff87-b67f-4b68-8106-e297566641ef.hana.prod-ap11.hanacloud.ondemand.com"

# Bedrock model configuration
MODEL_ID = "anthropic--claude-3.5-sonnet"


def load_config():
    """Load environment variables from .env file."""
    load_dotenv()

def get_documents_dir_path():
    """Get default documents directory path."""
    return os.path.join(load_config()['local_path'], "Documents")

def get_default_schema():
    return os.getenv('DEF_SCHEMA', 'TREASURY_AI')

DEF_SCHEMA = get_default_schema()    