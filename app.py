from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import boto3
import pyodbc
import json
import os
import pymssql
from langchain_community.embeddings import BedrockEmbeddings
from langchain.vectorstores import PGVector
from datetime import datetime
from typing import Optional, List, Tuple, Any

# AWS Bedrock client setup
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='ap-south-1'
)

# Database connection configurations
SQL_SERVER_CONN = {
    'driver': '{ODBC Driver 17 for SQL Server}',
    'server': os.getenv('SQL_SERVER_HOST'),
    'database': os.getenv('SQL_SERVER_DB'),
    'uid': os.getenv('SQL_SERVER_USER'),
    'pwd': os.getenv('SQL_SERVER_PASSWORD'),
    'Encrypt': 'yes',  # Added for security
    'TrustServerCertificate': 'yes'
}

def get_table_metadata():
    """Fetch table metadata using AWS Glue Catalog"""
    glue_client = boto3.client('glue')
    
    try:
        response = glue_client.get_tables(
            DatabaseName="sqlserver_metadata"
        )
        
        metadata = {}
        target_tables = ['xtstock2025']
        
        for table in response['TableList']:
            if table['Name'] in target_tables:
                metadata[table['Name']] = {
                    'columns': [col['Name'] for col in table['StorageDescriptor']['Columns']],
                    'description': table.get('Description', ''),
                    'column_types': {col['Name']: col['Type'] for col in table['StorageDescriptor']['Columns']}
                }
        
        if not metadata:
            st.warning("No target tables found in the Glue Catalog")
            
        return metadata
    except Exception as e:
        st.error(f"Error fetching metadata: {str(e)}")
        return None

def generate_sql_query(natural_language_query, metadata):
    """Generate SQL query using Claude-3 with Messages API"""

    bedrock = boto3.client("bedrock-runtime")

    messages = [
        {"role": "user", "content": f"""Given the following database schema:
        {json.dumps(metadata, indent=2)}

        Convert this natural language query to SQL:
        {natural_language_query}

        Rules:
        1. Use proper SQL Server syntax.
        2. Query from the `xtstock2025` table.
        3. Use column names exactly as specified in the schema.
        4. If filtering by date, use `stocksourcedate`.
        5. If filtering by stock quantity, use `stockqty`.
        6. Return only the SQL query without any explanation.
        
        Examples:

        **Example 1:**
        Natural language query: "Get all stock details for category 'Apparel'."
        SQL Query:
        ```
        SELECT * FROM xtstock2025 WHERE category = 'Apparel';
        ```

        **Example 2:**
        Natural language query: "Find total stock quantity for site 'AdhocFabric'."
        SQL Query:
        ```
        SELECT SUM(stockqty) AS total_stock FROM xtstock2025 WHERE sitename = 'AdhocFabric';
        ```

        **Example 3:**
        Natural language query: "Show all red articles in size 'M' with stock greater than 10."
        SQL Query:
        ```
        SELECT * FROM xtstock2025 
        WHERE colorname = 'Red' AND sizename = 'M' AND stockqty > 10;
        ```

        **Example 4:**
        Natural language query: "Get stock value for 'Blue' articles older than 30 days."
        SQL Query:
        ```
        SELECT SUM(stockvalue) AS total_stock_value FROM xtstock2025 
        WHERE colorname = 'Blue' AND stockageindays > 30;
        ```

        **Example 5:**
        Natural language query: "Find stock added after January 1, 2024."
        SQL Query:
        ```
        SELECT * FROM xtstock2025 WHERE stocksourcedate > '2024-01-01';
        ```

        Now generate the SQL query based on the given natural language query."""}
    ]

    try:
        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": messages,
                "max_tokens": 500,
                "temperature": 0.1,
                "top_p": 0.9
            }).encode('utf-8')
        )

        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text'].strip()

    except Exception as e:
        st.error(f"Error generating SQL: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

def execute_sql_query(query: str, params: Optional[Tuple] = None) -> List[Tuple[Any, ...]]:
    """
    Execute a SQL query using pymssql and return the results.
    
    Args:
        query (str): The SQL query to execute
        params (tuple, optional): Parameters to be used with the query for prepared statements
        
    Returns:
        List[Tuple]: Results of the query
        
    Raises:
        Exception: If there's an error connecting to the database or executing the query
    """
    try:
        # Get database connection details from environment variables
        server = SQL_SERVER_CONN.get('server')
        database = SQL_SERVER_CONN.get('database')
        username = SQL_SERVER_CONN.get('uid')
        password = SQL_SERVER_CONN.get('pwd')
        
        # Validate connection parameters
        if not all([server, database, username, password]):
            raise ValueError("Missing database connection parameters")
            
        # Establish connection
        conn = pymssql.connect(
            server=server,
            database=database,
            user=username,
            password=password
        )
        
        try:
            with conn.cursor() as cursor:
                # Execute query with or without parameters
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)

                # Get column names from cursor description
                column_names = [column[0] for column in cursor.description]
                
                # Fetch all results
                results = cursor.fetchall()

                # Convert results to list of dictionaries
                results_with_columns = [dict(zip(column_names, row)) for row in results]
                
                # Commit if the query modified data
                if query.lower().strip().startswith(('insert', 'update', 'delete')):
                    conn.commit()
                    
                return results_with_columns
                
        except Exception as e:
            conn.rollback()  # Rollback in case of error
            raise Exception(f"Error executing query: {str(e)}")
            
        finally:
            conn.close()
            
    except Exception as e:
        raise Exception(f"Database connection error: {str(e)}")

def main():

    st.set_page_config(page_title="Text to SQL Converter", page_icon=":zap:", layout="wide")
    
    st.title("Text to SQL Converter")
    st.markdown("Convert natural language questions to SQL queries")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_input = st.text_area(
            "Enter your question:",
            placeholder="e.g., Show all orders from xtBuyerOrder table",
            height=100
        )
    
    with col2:
        execute_query = st.checkbox("Execute query after generation", value=True)
        submit_button = st.button("Generate SQL", use_container_width=True)
    
    # Results section
    if submit_button and user_input:
        with st.spinner("Processing..."):
            sql_query = generate_sql_query(user_input, st.session_state.metadata)

            if sql_query:
                st.subheader("Generated SQL Query:")
                st.code(sql_query, language="sql")
                
                if execute_query:
                    # Execute query
                    results = execute_sql_query(sql_query)
                    if results is not None:
                        st.subheader("Query Results:")
                        st.dataframe(
                            results,
                            use_container_width=True,
                            hide_index=True
                        )

if __name__ == "__main__":
    main()
