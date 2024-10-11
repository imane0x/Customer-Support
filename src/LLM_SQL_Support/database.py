# src/database.py

import sqlite3
from config import RAW_DATA_PATH

def connect_database(db_path=RAW_DATA_PATH):
    conn = sqlite3.connect(db_path)
    return conn
