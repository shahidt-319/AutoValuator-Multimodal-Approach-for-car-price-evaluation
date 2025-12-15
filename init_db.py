import sqlite3
import os
import json

DB_PATH = os.path.join(os.path.dirname(__file__), 'autovaluator.db')

def get_connection():
    """Return a new SQLite connection with foreign keys enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def create_tables(cursor):
    """Create all needed tables if they don't exist yet."""
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE,
            phone TEXT UNIQUE,
            password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            otp TEXT,
            otp_verified INTEGER DEFAULT 0
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cars (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL,
            year INTEGER,
            km_driven INTEGER,
            owner TEXT,
            fuel TEXT,
            seller TEXT,
            transmission TEXT,
            location TEXT,
            registration_state TEXT,
            engine_condition TEXT,
            email TEXT,
            car_name TEXT,
            predicted_price REAL DEFAULT 0,
            seller_price REAL DEFAULT 0,
            cleaned_audio TEXT,
            noise_audio TEXT,
            clipped_audio TEXT,
            vin_details TEXT    -- VIN details stored as JSON
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS car_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            car_id INTEGER NOT NULL,
            photo TEXT,
            damage_status TEXT,
            FOREIGN KEY(car_id) REFERENCES cars(id) ON DELETE CASCADE
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cart (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL,
            car_id INTEGER NOT NULL,
            car_name TEXT,
            fuel TEXT,
            transmission TEXT,
            location TEXT,
            photo TEXT,
            seller_price REAL DEFAULT 0,
            FOREIGN KEY(car_id) REFERENCES cars(id) ON DELETE CASCADE
        );
    ''')

def add_column_if_missing(cursor, table, column, ctype):
    """Add a column if it's missing in the given table."""
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [info[1] for info in cursor.fetchall()]
    if column not in columns:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ctype};")
        print(f"✅ Added column '{column}' to '{table}'.")

def check_vin_json_validity(conn):
    """Check VIN JSON validity and optionally fix malformed JSON."""
    c = conn.cursor()
    rows = c.execute("SELECT id, car_name, vin_details FROM cars WHERE vin_details IS NOT NULL").fetchall()

    if not rows:
        print("\nℹ️ No VIN details found in database yet.")
        return

    print("\n🔎 Checking VIN JSON validity for recent cars...")
    invalid_entries = []
    for car_id, car_name, vin_data in rows:
        try:
            json.loads(vin_data)
        except Exception as e:
            invalid_entries.append((car_id, car_name, vin_data, str(e)))

    if not invalid_entries:
        print("✅ All VIN details are valid JSON!\n")
        return

    print(f"⚠️ Found {len(invalid_entries)} invalid VIN JSON entries:")
    for car_id, car_name, vin_data, err in invalid_entries:
        print(f"   • Car ID {car_id} ({car_name}): {err}")
        if "'" in vin_data:
            print("     → Suspect single-quoted JSON format.")

    # Ask user if they want to auto-fix
    choice = input("\nWould you like to auto-fix single-quoted VIN JSON entries? (y/n): ").strip().lower()
    if choice == 'y':
        fixed_count = 0
        for car_id, car_name, vin_data, _ in invalid_entries:
            if "'" in vin_data:
                fixed = vin_data.replace("'", '"')
                try:
                    json.loads(fixed)
                    c.execute("UPDATE cars SET vin_details=? WHERE id=?", (fixed, car_id))
                    fixed_count += 1
                except Exception:
                    print(f"❌ Could not fix VIN JSON for Car ID {car_id}")
        conn.commit()
        print(f"\n✅ Fixed {fixed_count} VIN entries successfully.\n")
    else:
        print("\n⚠️ Skipped auto-fix. VIN JSON remains unchanged.\n")

def ensure_admin_account(cursor, target_email):
    """Ensure the given email is present and set to role 'admin'."""
    user = cursor.execute("SELECT id FROM users WHERE email=?", (target_email,)).fetchone()
    if user:
        cursor.execute("UPDATE users SET role='admin' WHERE email=?", (target_email,))
        print(f"🔐 Set role='admin' for user {target_email}")
    else:
        # Insert a default user -- you may want to set/reset/change the password as needed
        cursor.execute(
            "INSERT INTO users (username, email, password, role, otp_verified) VALUES (?, ?, ?, ?, 1)",
            ("admin", target_email, "admin_password_change_me", "admin")
        )
        print(f"🔐 Inserted admin user for {target_email}")

def initialize_database():
    """Initialize or update the database schema and check VIN JSON."""
    conn = get_connection()
    c = conn.cursor()

    create_tables(c)
    print("\n🔍 Checking/updating for any missing columns...")
    expected_cols = [
        ('cars', 'engine_condition', 'TEXT'),
        ('cars', 'cleaned_audio', 'TEXT'),
        ('cars', 'noise_audio', 'TEXT'),
        ('cars', 'clipped_audio', 'TEXT'),
        ('cars', 'vin_details', 'TEXT'),
        ('users', 'otp', 'TEXT'),
        ('users', 'otp_verified', 'INTEGER DEFAULT 0'),
    ]

    for table, col, ctype in expected_cols:
        add_column_if_missing(c, table, col, ctype)

    conn.commit()
    print("\n✅ Database schema verified.")

    ensure_admin_account(c, "shahiddar763312Aa@gmail.com")
    conn.commit()

    # Run VIN JSON validity check
    check_vin_json_validity(conn)

    conn.close()
    print("🚀 Initialization complete — Database ready!\n")

if __name__ == '__main__':
    initialize_database()
