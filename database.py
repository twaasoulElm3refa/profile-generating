import mysql.connector
from mysql.connector import Error
from datetime import timedelta
from dotenv import load_dotenv
import os

load_dotenv()  # يبحث عن .env في مجلد المشروع الحالي

api_key = os.getenv("OPENAI_API_KEY")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_name = os.getenv("DB_NAME")



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

def fetch_press_releases(user_id: str ):
    connection = get_db_connection()
    if connection is None:
        print("Failed to establish database connection")
        return []
    
    try:
        cursor = connection.cursor(dictionary=True)
        query = """
        SELECT * 
        FROM wpl3_press_release_Form
        WHERE user_id = %s 
        """
        cursor.execute(query, (user_id,))

        # Fetch the first row
        all_user_articles = cursor.fetchall()

        return all_user_articles

    except Error as e:
        print(f"Error fetching data: {e}")
        return []
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()



def update_press_release(user_id, organization_name, article):
    connection = get_db_connection()
    if connection is None:
        return False
    
    try:
        cursor = connection.cursor()
        query = """
        INSERT INTO wpl3_articles (user_id, organization_name, article)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE article = VALUES(article)
        """
        cursor.execute(query, (user_id, organization_name, article))
        connection.commit()
        return True
    except Error as e:
        print(f"Error updating data: {e}")
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
