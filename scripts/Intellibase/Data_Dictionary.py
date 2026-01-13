import os
import typing
from hdbcli.dbapi import Connection
import pandas as pd
from logger_setup import get_logger
import csv
import json
from db_connection import get_db_connection, release_db_connection
from env_config import TREASURY_TABLE
# Configure logger
logger = get_logger()
from destination_srv import get_destination_service_credentials, generate_token, fetch_destination_details, extract_hana_credentials

# Type aliases for consistency
DataFrameType = pd.DataFrame


# Initialize HANA Credentials
logger.info("====> DANS_Upload.py -> GET HANA CREDENTIALS <====")
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


class DictionaryUpdater:
    """
    Handles reading and updating the Dictionary sheet with unique values from HANA DB.
    
    Responsibilities:
    - Read Dictionary sheet from Excel
    - Add new column for unique values
    - Update specific rows with unique values from HANA DB table
    - Save as text file for LLM input
    """
    Connection = get_db_connection()
    @staticmethod
    def read_dictionary_sheet(excel_path: str, sheet_name: str = "Dictionary") -> DataFrameType:
        """
        Read the Dictionary sheet from the Excel file with error handling.
        
        Args:
            excel_path (str): Path to Excel file
            sheet_name (str): Name of the sheet to read (default: Dictionary)
        
        Returns:
            Loaded DataFrame
        
        Raises:
            FileNotFoundError: If Excel file doesn't exist
            ValueError: If sheet not found or file is empty
        """
        if not os.path.exists(excel_path):
            logger.error(f"Excel file not found at: {excel_path}")
            raise FileNotFoundError(f"Excel file not found at: {excel_path}")
        
        try:
            logger.info(f"Reading Dictionary sheet from: {excel_path}")
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            
            if df.empty:
                raise ValueError(f"No data found in sheet '{sheet_name}'")
            
            # Verify expected columns
            expected_cols = ["COLUMN", "DESCRIPTION", "LONG DESCRIPTION"]
            if not all(col in df.columns for col in expected_cols):
                raise ValueError(f"Dictionary sheet missing required columns. Expected: {expected_cols}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error reading Dictionary sheet: {e}")
            raise
    
    @staticmethod
    def get_unique_values_from_hana(connection, column_name: str, table_name: str = TREASURY_TABLE) -> list:
        """
        Get unique values for a specific column from HANA DB table.
        
        Args:
            connection: Database connection object
            column_name (str): Name of the column to get unique values for
            table_name (str): Name of the HANA DB table (default: TREASURY_TABLE)
        
        Returns:
            list: List of unique values for the column
        
        Raises:
            Exception: If database query fails
        """
        try:
            logger.info(f"Fetching unique values for column {column_name} from {table_name}")
            
            cursor = connection.cursor()
            
            # Query to get distinct values, excluding NULL values
            query = f"""
                SELECT DISTINCT {column_name}
                FROM {table_name}
                WHERE {column_name} IS NOT NULL
                AND {column_name} != ''
                ORDER BY {column_name}
            """
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            # Extract values from query results
            unique_values = [str(row[0]).strip() for row in results if row[0] is not None and str(row[0]).strip()]
            
            cursor.close()
            
            logger.info(f"Found {len(unique_values)} unique values for column {column_name}")
            return unique_values
            
        except Exception as e:
            logger.error(f"Error fetching unique values for column {column_name}: {e}")
            raise
    
    @staticmethod
    def update_dictionary_with_hana_data(
        dictionary_df: DataFrameType, 
        db_connection,
        unique_value_col: str = "UNIQUE_VALUES",
        table_name: str = TREASURY_TABLE
    ) -> DataFrameType:
        """
        Add a new column to the Dictionary DataFrame and update unique values from HANA DB.
        
        Args:
            dictionary_df (pd.DataFrame): Dictionary sheet DataFrame
            db_connection: Database connection object
            unique_value_col (str): Name of the new column for unique values
            table_name (str): Name of the HANA DB table
        
        Returns:
            Updated Dictionary DataFrame
        """
        # Create a copy to avoid modifying the original
        updated_df = dictionary_df.copy()
        
        # Initialize the new column with None
        updated_df[unique_value_col] = None
        
        # Define columns to update
        columns_to_update = {
            "GLB_ASSET_CLASS_2": "GLB_ASSET_CLASS_2",
            "GLB_COUPON_TYPE": "GLB_COUPON_TYPE",
            "GLB_FINAL_CCY": "GLB_FINAL_CCY",
            "GLB_FINAL_COUNTRY_NAME": "GLB_FINAL_COUNTRY_NAME",
            "GLB_FV_HTC": "GLB_FV_HTC",
            "GLB_GLOBAL_REGION": "GLB_GLOBAL_REGION",
            "GLB_HIGH_LVL_STRATEGY": "GLB_HIGH_LVL_STRATEGY",
            "GLB_HQLA": "GLB_HQLA",
            "GLB_PRODUCT_SUBTYPE": "GLB_PRODUCT_SUBTYPE",
            "GLB_SOLO_SUB": "GLB_SOLO_SUB"
        }
        
        for dict_col, db_col in columns_to_update.items():
            try:
                # Get unique values from HANA DB
                unique_values = DictionaryUpdater.get_unique_values_from_hana(
                    db_connection, db_col, table_name
                )
                
                if not unique_values:
                    logger.warning(f"No unique values found for column {db_col} in {table_name}")
                    continue
                
                # Convert list to comma-separated string
                unique_values_str = ", ".join(unique_values)
                
                # Find the row in dictionary_df where COLUMN matches dict_col
                mask = updated_df["COLUMN"] == dict_col
                if not mask.any():
                    logger.warning(f"Column {dict_col} not found in Dictionary sheet")
                    continue
                
                # Update the UNIQUE_VALUES column for this row
                updated_df.loc[mask, unique_value_col] = unique_values_str
                logger.info(f"Updated unique values for {dict_col}: {len(unique_values)} values")
                logger.debug(f"Values for {dict_col}: {unique_values_str[:200]}{'...' if len(unique_values_str) > 200 else ''}")
            
            except Exception as e:
                logger.error(f"Error updating unique values for {dict_col}: {e}")
                continue
        
        # Clean data: Replace tab characters with spaces in all columns
        for col in updated_df.columns:
            updated_df[col] = updated_df[col].astype(str).str.replace('\t', ' ', regex=False)
        
        return updated_df
    
    @staticmethod
    def save_updated_dictionary_csv(df: DataFrameType, output_path: str) -> None:
        """
        Save the updated Dictionary DataFrame to a text file (CSV format with tab delimiter) for LLM input.
        
        Args:
            df (pd.DataFrame): Updated Dictionary DataFrame
            output_path (str): Path to save the output text file
        
        Raises:
            Exception: If saving fails
        """
        try:
            logger.info(f"Saving updated Dictionary to: {output_path}")
            # Save as CSV with tab delimiter, quoting all fields
            df.to_csv(output_path, sep='\t', index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
            logger.info(f"Successfully saved updated Dictionary to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save updated Dictionary to {output_path}: {e}")
            raise

    @staticmethod
    def save_updated_dictionary(df: DataFrameType, output_path: str) -> None:
        """
        Save the updated Dictionary DataFrame to a JSON file for LLM input.
        
        Args:
            df (pd.DataFrame): Updated Dictionary DataFrame
            output_path (str): Path to save the output JSON file
        
        Raises:
            Exception: If saving fails
        """
        try:
            logger.info(f"Saving updated Dictionary (JSON format) to: {output_path}")
            
            # Convert DataFrame to list of dictionaries for better JSON structure
            data_dict = df.to_dict('records')
            
            # Handle NaN values by converting them to None
            for record in data_dict:
                for key, value in record.items():
                    if pd.isna(value) or value == 'nan':
                        record[key] = None
            
            # Create a structured JSON with metadata
            json_output = {
                "metadata": {
                    "description": "Data Dictionary with unique values from HANA DB",
                    "total_columns": len(data_dict),
                    "generated_timestamp": pd.Timestamp.now().isoformat(),
                    "columns": list(df.columns.tolist()),
                    "data_source": TREASURY_TABLE
                },
                "dictionary_data": data_dict
            }
            
            # Save as JSON with proper formatting
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_output, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Successfully saved updated Dictionary (JSON format) to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save updated Dictionary (JSON format) to {output_path}: {e}")
            raise        

def process_dictionary_sheet(
    excel_path: str, 
    output_path: str, 
    db_connection,
    table_name: str = TREASURY_TABLE
) -> None:
    """
    Orchestrate the process of updating the Dictionary sheet with unique values from HANA DB.
    
    Args:
        excel_path (str): Path to the input Excel file
        output_path (str): Path to save the updated Dictionary text file
        db_connection: Database connection object
        table_name (str): Name of the HANA DB table (default: TREASURY_TABLE)
    
    Raises:
        Exception: If any step in the process fails
    """
    try:
        # 1. Read Dictionary sheet
        dictionary_df = DictionaryUpdater.read_dictionary_sheet(excel_path)
        
        # 2. Update Dictionary with unique values from HANA DB
        updated_dictionary = DictionaryUpdater.update_dictionary_with_hana_data(
            dictionary_df, db_connection, table_name=table_name
        )
        
        # 3. Save updated Dictionary as JSON file
        DictionaryUpdater.save_updated_dictionary(updated_dictionary, output_path)
        
        logger.info("Dictionary sheet processing with HANA DB completed successfully")
    
    except Exception as e:
        logger.error(f"Dictionary sheet processing with HANA DB failed: {e}")
        raise
