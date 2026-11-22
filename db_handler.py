import sqlite3
import hashlib
from datetime import datetime
import pandas as pd

DB_FILE = "soil_system.db"

def init_db():
    """Initializes the SQLite database with users and soil history tables."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # 1. Users Table (for Register / Login flow)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT NOT NULL,
        farm_location TEXT,
        created_at TEXT NOT NULL
    )
    """)

    # 2. Recommendation History Table (as shown in PPT Architecture)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS soil_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        n REAL NOT NULL,
        p REAL NOT NULL,
        k REAL NOT NULL,
        ph REAL NOT NULL,
        moisture REAL NOT NULL,
        temperature REAL NOT NULL,
        humidity REAL NOT NULL,
        soil_health TEXT NOT NULL,
        fertilizer TEXT NOT NULL,
        tested_at TEXT NOT NULL,
        FOREIGN KEY (username) REFERENCES users(username)
    )
    """)

    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def register_user(username, password, full_name, farm_location=""):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash, full_name, farm_location, created_at) VALUES (?, ?, ?, ?, ?)",
            (username.strip().lower(), hash_password(password), full_name.strip(), farm_location.strip(), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        return True, "Registration successful! You can now login."
    except sqlite3.IntegrityError:
        return False, "Username already exists. Please choose another username."
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT full_name, farm_location FROM users WHERE username = ? AND password_hash = ?",
        (username.strip().lower(), hash_password(password))
    )
    user = cursor.fetchone()
    conn.close()
    if user:
        return True, {"username": username.strip().lower(), "full_name": user[0], "farm_location": user[1]}
    return False, "Invalid username or password."

def save_recommendation_history(username, n, p, k, ph, moisture, temp, humidity, health, fertilizer):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO soil_history (username, n, p, k, ph, moisture, temperature, humidity, soil_health, fertilizer, tested_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (username, n, p, k, ph, moisture, temp, humidity, health, fertilizer, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def get_user_history(username):
    conn = sqlite3.connect(DB_FILE)
    query = """
    SELECT n, p, k, ph, moisture, temperature, humidity, soil_health, fertilizer, tested_at
    FROM soil_history
    WHERE username = ?
    ORDER BY id DESC
    """
    df = pd.read_sql_query(query, conn, params=(username,))
    conn.close()
    return df

# Initialize database on import
init_db()
