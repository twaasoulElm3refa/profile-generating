import mysql.connector
from mysql.connector import Error
from datetime import timedelta
from dotenv import load_dotenv
import os

load_dotenv()  # يبحث عن .env في مجلد المشروع الحالي

db_host = os.getenv("db_host")
db_port = os.getenv("db_port")
db_user = os.getenv("db_user")
db_password = os.getenv("db_password")
db_name = os.getenv("db_name")


def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_password,
            port=db_port
        )

        if connection.is_connected():
            print("✅ Connected!")
            return connection
    except Error as e:
        print("❌ Failed.")
        print(f"Error connecting to MySQL: {e}")
        return None

def fetch_profile_data(user_id: str ):
    connection = get_db_connection()
    if connection is None:
        print("Failed to establish database connection")
        return []
    
    try:
        cursor = connection.cursor(dictionary=True)
        query = """
        SELECT * 
        FROM wpl3_profile_generating_tool
        WHERE user_id = %s 
        """
        cursor.execute(query, (user_id,))

        # Fetch the first row
        all_profiles_content = cursor.fetchall()

        return all_profiles_content

    except Error as e:
        print(f"Error fetching data: {e}")
        return []
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()



def insert_generated_profile(user_id, organization_name, generated_profile, input_type='Using FORM'):
    connection = get_db_connection()
    if connection is None:
        return False
    
    try:
        cursor = connection.cursor()
        query = """
        INSERT INTO wpl3_profile_result (user_id, organization_name, generated_profile,input_type)
        VALUES (%s, %s, %s , %s)
        ON DUPLICATE KEY UPDATE generated_profile = VALUES(generated_profile)
        """
        cursor.execute(query, (user_id, organization_name, generated_profile,input_type))
        connection.commit()
        return True
    except Error as e:
        print(f"Error updating data: {e}")
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()