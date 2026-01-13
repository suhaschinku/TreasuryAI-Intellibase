import os
import typing
import numpy as np
import pandas as pd

from env_config import TREASURY_TABLE
# Import custom modules
from db_connection import get_db_connection, release_db_connection
from logger_setup import get_logger
from Intellibase.Data_Dictionary import process_dictionary_sheet  # Import the dictionary function
from destination_srv import get_destination_service_credentials, generate_token, fetch_destination_details, extract_hana_credentials

# Configure logger
logger = get_logger()

# Initialize HANA Credentials using CredentialsManager
logger.info("====> DANS_Upload.py -> GET HANA CREDENTIALS <====")

# Import CredentialsManager using parent module reference
import sys
# Add parent directory to path only if running as script (not when imported as module)
if __name__ == "__main__" or 'scripts' not in sys.path[0]:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

from credentials_manager import credentials_manager

# Initialize credentials if not already done
credentials_manager.initialize()

# Get HANA credentials from singleton
HANA_CREDENTIALS = credentials_manager.get_hana_credentials()
logger.info(f"HANA Credentials loaded from CredentialsManager: {HANA_CREDENTIALS}")

# Type aliases for improved type hinting
DataFrameType = pd.DataFrame
DatabaseConnectionType = typing.Any
DatabaseCursorType = typing.Any

class DANSTableSchemaValidator:
    """
    Handles schema validation for the DANS Treasury Fact Table.
    
    Responsibilities:
    - Define table schema
    - Validate incoming Excel file structure
    - Process and transform data to match schema requirements
    """
    
    # Define table schema with column names and their SQL data types
    TABLE_SCHEMA = {
        # String columns
        "GLB_ASSET_CLASS_2": "VARCHAR(100)",
        "GLB_COUPON_TYPE": "VARCHAR(100)",
        "GLB_CUSIP": "VARCHAR(100)",
        "GLB_INSTRUMENT": "VARCHAR(100)",
        "GLB_ISIN": "VARCHAR(100)",
        "GLB_FINAL_CCY": "VARCHAR(100)",
        "GLB_FINAL_COUNTRY_NAME": "VARCHAR(100)",
        "GLB_FV_HTC": "VARCHAR(100)",
        "GLB_GLOBAL_REGION": "VARCHAR(100)",
        "GLB_HIGH_LVL_STRATEGY": "VARCHAR(100)",
        "GLB_HQLA": "VARCHAR(100)",
        "GLB_INDEX_NAME": "VARCHAR(100)",
        "GLB_ISSUE_RATING_SP": "VARCHAR(100)",
        "GLB_ISSUER_NAME": "VARCHAR(100)",
        "GLB_MAP_PORTFOLIO": "VARCHAR(100)",
        "GLB_SOLO_SUB": "VARCHAR(100)",
        "GLB_PRODUCT_SUBTYPE": "VARCHAR(100)",
        
        # Date columns
        "GLB_LAST_RESET_DATE": "Date",
        "GLB_MATURITY_DATE": "Date",
        "GLB_REPORT_DATE": "Date",
        
        # Numeric columns
        "GLB_ASW_DM": "DECIMAL(38,8)",
        "GLB_BOOK_PRICE": "DECIMAL(38,2)",
        "GLB_BOOK_VALUE_USD": "DECIMAL(38,2)",
        "GLB_CR_DELTA_TOTAL": "DECIMAL(38,2)",
        "GLB_IR_PV01_TOTAL": "DECIMAL(38,2)",
        "GLB_MARKET_PRICE": "DECIMAL(38,2)",
        "GLB_MARKET_VALUE_USD": "DECIMAL(38,2)",
        "GLB_MARKET_YIELD": "DECIMAL(38,8)",
        "GLB_MTM_USD": "DECIMAL(38,2)",
        "GLB_NOTIONAL_USD": "DECIMAL(38,2)",
        "GLB_RWA": "DECIMAL(38,2)",
        "GLB_YIELD_IMPACT": "DECIMAL(38,8)"
    }
    
    @classmethod
    def get_expected_columns(cls) -> typing.List[str]:
        """
        Retrieve the list of expected columns.
        
        Returns:
            List of column names
        """
        return list(cls.TABLE_SCHEMA.keys())
    
    @classmethod
    def validate_schema(cls, df: DataFrameType) -> typing.Dict[str, typing.Any]:
        """
        Validate the schema of the input DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame to validate
        
        Returns:
            Dict with validation results
        """
        # Check column names
        expected_columns = cls.get_expected_columns()
        missing_cols = set(expected_columns) - set(df.columns)
        extra_cols = set(df.columns) - set(expected_columns)
        
        # Prepare validation report
        validation_result = {
            "is_valid": len(missing_cols) == 0,
            "missing_columns": list(missing_cols),
            "extra_columns": list(extra_cols)
        }
        
        # Log validation details
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
        
        if extra_cols:
            logger.warning(f"Extra columns found: {extra_cols}")
        
        return validation_result
    
    @classmethod
    def process_data(cls, df: DataFrameType) -> DataFrameType:
        """
        Process dataframe to match table schema with 2 decimal places rounding.
        
        Args:
            df (pd.DataFrame): Input DataFrame to process
        
        Returns:
            Processed DataFrame
        """
        # Make a copy to avoid modifying the original dataframe
        processed_df = df.copy()
        
        # Process each column based on its type
        for col, col_type in cls.TABLE_SCHEMA.items():
            if col not in processed_df.columns:
                logger.warning(f"Column {col} not found in DataFrame")
                continue
            
            if col_type.startswith("VARCHAR"):
                # For string columns, convert NaN to None and truncate
                max_length = int(col_type.split("(")[1].strip(")"))
                processed_df[col] = processed_df[col].apply(
                    lambda x: None if pd.isna(x) else str(x)[:max_length]
                )
            
            elif col_type == "Date":
                # Convert to datetime format, leaving NaT as is
                processed_df[col] = pd.to_datetime(processed_df[col], errors="coerce")
            
            elif col_type.startswith("DECIMAL"):
                # Extract scale from DECIMAL(p,s)
                scale = int(col_type.split(",")[1].strip(")"))
                
                # Replace NaN, inf, -inf with None
                mask_invalid = ~np.isfinite(pd.to_numeric(processed_df[col], errors='coerce'))
                if mask_invalid.any():
                    logger.warning(f"Column {col}: Replacing {mask_invalid.sum()} non-finite values (NaN, inf) with NULL")
                    processed_df.loc[mask_invalid, col] = None
                
                # Convert remaining values to numeric and round
                processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce").round(scale)
        
        return processed_df


class DANSDataLoader:
    """
    Handles data loading, transformation, and insertion into database.
    
    Responsibilities:
    - Read Excel files
    - Prepare data for database insertion
    - Perform batch insertions
    """
    
    @staticmethod
    def read_excel(excel_path: str, sheet_name: str = "Data") -> DataFrameType:
        """
        Read Excel file with error handling.
        
        Args:
            excel_path (str): Path to Excel file
            sheet_name (str, optional): Name of the sheet to read
        
        Returns:
            Loaded DataFrame
        
        Raises:
            FileNotFoundError: If Excel file doesn't exist
            ValueError: If sheet not found or file is empty
        """
        # Validate file existence
        if not os.path.exists(excel_path):
            logger.error(f"Excel file not found at: {excel_path}")
            raise FileNotFoundError(f"Excel file not found at: {excel_path}")
        
        try:
            # Read Excel file
            logger.info(f"Reading Excel file from: {excel_path}")
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            
            # Check if DataFrame is empty
            if df.empty:
                raise ValueError(f"No data found in sheet '{sheet_name}'")
            
            return df
        
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            raise
    
    @staticmethod
    def prepare_records_for_insertion(df: DataFrameType) -> typing.List[typing.Tuple]:
        """
        Convert DataFrame to a list of records for database insertion.
        
        Args:
            df (pd.DataFrame): Processed DataFrame
        
        Returns:
            List of records ready for insertion
        """
        records = []
        for _, row in df.iterrows():
            record = []
            for val in row:
                if pd.isna(val) or val is None or (isinstance(val, float) and not np.isfinite(val)):
                    record.append(None)
                elif isinstance(val, pd.Timestamp):
                    record.append(val.strftime("%Y-%m-%d"))
                else:
                    record.append(val)
            records.append(tuple(record))
        
        return records


class DANSTableInserter:
    """
    Handles database connection and data insertion operations.
    
    Responsibilities:
    - Manage database connections
    - Execute table truncation
    - Perform batch insertions
    - Handle transactions
    """
    
    @staticmethod
    def insert_data(
        records: typing.List[typing.Tuple], 
        schema_name: str = "TREASURY_AI", 
        table_name: str = TREASURY_TABLE
    ) -> None:
        """
        Insert records into the specified database table.
        
        Args:
            records (List[Tuple]): Processed records to insert
            schema_name (str): Database schema name
            table_name (str): Target table name
        
        Raises:
            Exception: If insertion fails
        """
        conn = None
        cursor = None
        
        try:
            # Get database connection
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Prepare SQL statements
            expected_columns = DANSTableSchemaValidator.get_expected_columns()
            columns_str = ", ".join([f'"{col}"' for col in expected_columns])
            placeholders = ", ".join(["?" for _ in expected_columns])
            
            insert_query = f'INSERT INTO "{schema_name}"."{table_name}" ({columns_str}) VALUES ({placeholders})'
            truncate_query = f'TRUNCATE TABLE "{schema_name}"."{table_name}"'
            
            try:
                # Disable auto-commit
                conn.setautocommit(False)
                
                # Truncate existing table data
                cursor.execute(truncate_query)
                logger.info(f"Existing data in {schema_name}.{table_name} truncated")
                
                # Insert records in batches
                batch_size = 1000
                total_records = len(records)
                
                for i in range(0, total_records, batch_size):
                    batch = records[i:i + batch_size]
                    try:
                        cursor.executemany(insert_query, batch)
                        logger.info(f"Inserted batch {i//batch_size + 1}/{(total_records-1)//batch_size + 1}: {len(batch)} records")
                    except Exception as batch_error:
                        logger.error(f"Error inserting batch starting at row {i}: {batch_error}")
                        # Fallback to single record insertion for detailed error tracking
                        for j, record in enumerate(batch):
                            try:
                                cursor.execute(insert_query, record)
                            except Exception as record_error:
                                logger.error(f"Error at row {i+j}: {record_error}")
                                logger.error(f"Problematic record: {record}")
                                raise
                
                # Commit the transaction
                conn.commit()
                logger.info(f"Successfully inserted {total_records} records into {schema_name}.{table_name}")
            
            except Exception as insertion_error:
                # Rollback the transaction if any error occurs
                conn.rollback()
                logger.error(f"Transaction rolled back. Error during data insertion: {insertion_error}")
                raise
            
            finally:
                # Restore auto-commit to default
                conn.setautocommit(True)
        
        except Exception as e:
            logger.error(f"Data insertion failed: {e}")
            raise
        
        finally:
            # Ensure resources are properly closed
            if cursor:
                cursor.close()
            if conn:
                release_db_connection(conn)


def insert_excel_to_hana(
    excel_path: str, 
    schema_name: str = "TREASURY_AI", 
    table_name: str = TREASURY_TABLE
) -> None:
    """
    Orchestrate the entire data insertion process, process dictionary sheet, and delete the Excel file.
    
    Args:
        excel_path (str): Path to the Excel file
        schema_name (str): Database schema name
        table_name (str): Target table name
    
    Raises:
        Exception: If any step in the process fails
    """
    localpath = os.getenv('LOCALPATH', os.getcwd())
    dictionary_path = os.path.join(localpath, "Intellibase", "Data_Dictionary.txt")
    
    try:
        # 1. Read Excel file
        df = DANSDataLoader.read_excel(excel_path)
        
        # 2. Validate schema
        validation_result = DANSTableSchemaValidator.validate_schema(df)
        if not validation_result['is_valid']:
            raise ValueError(f"Schema validation failed. Missing columns: {validation_result['missing_columns']}")
        
        # 3. Process data
        processed_df = DANSTableSchemaValidator.process_data(df[DANSTableSchemaValidator.get_expected_columns()])
        
        # 4. Remove rows where all values are NULL
        valid_rows = ~processed_df.isna().all(axis=1)
        if not valid_rows.any():
            raise ValueError("No valid rows to insert after processing")
        
        processed_df = processed_df[valid_rows]
        logger.info(f"Processing complete. {len(processed_df)} valid rows ready for insertion.")
        
        # 5. Prepare records
        records = DANSDataLoader.prepare_records_for_insertion(processed_df)
        
        # 6. Insert data
        DANSTableInserter.insert_data(records, schema_name, table_name)
        
        # 7. Process dictionary sheet
        logger.info("Starting dictionary sheet processing")
        process_dictionary_sheet(excel_path, dictionary_path)
        logger.info("Dictionary sheet processing completed")
    
    except Exception as e:
        logger.error(f"Excel to HANA insertion or dictionary processing failed: {e}")
        raise
    
    finally:
        # Delete the Excel file regardless of success or failure
        try:
            os.remove(excel_path)
            logger.info(f"Successfully deleted Excel file: {excel_path}")
        except Exception as delete_error:
            logger.error(f"Failed to delete Excel file {excel_path}: {delete_error}")
            # Log the error but don't raise to ensure the function completes
