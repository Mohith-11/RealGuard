import json
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from app.config import (
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
)


def get_db_connection():
    """Establish and return a connection to PostgreSQL database."""
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )
    return conn


def init_db():
    """Create the predictions table if it does not exist."""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS predictions (
        id                   SERIAL PRIMARY KEY,
        transaction_time     DOUBLE PRECISION,
        amount               DOUBLE PRECISION,
        prediction           INTEGER,
        label                VARCHAR(20),
        fraud_probability    DOUBLE PRECISION,
        model_version        INTEGER,
        created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        transaction_features JSONB,
        actual_label         SMALLINT,
        labeled_at           TIMESTAMP,
        CONSTRAINT valid_actual_label CHECK (actual_label IN (0, 1) OR actual_label IS NULL)
    );
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(create_table_query)
        conn.commit()
        print("PostgreSQL table 'predictions' initialized successfully.")
    except Exception as e:
        print(f"Error initializing database table: {e}")
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()


def migrate_db():
    """Idempotently add Phase 13/14 columns to existing predictions table."""
    migrations = [
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS transaction_features JSONB;",
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS actual_label SMALLINT;",
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS labeled_at TIMESTAMP;",
    ]
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            for sql in migrations:
                cursor.execute(sql)
            # Add constraint only if it does not already exist
            cursor.execute("""
                DO $body$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname = 'valid_actual_label'
                    ) THEN
                        ALTER TABLE predictions
                        ADD CONSTRAINT valid_actual_label
                        CHECK (actual_label IN (0, 1) OR actual_label IS NULL);
                    END IF;
                END
                $body$;
            """)
        conn.commit()
        print("Migration complete: all Phase 13/14 columns present.")
    except Exception as e:
        print(f"Migration warning: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


def insert_prediction(
    transaction_time: float,
    amount: float,
    prediction: int,
    label: str,
    fraud_probability: float,
    model_version: int,
    created_at: datetime = None,
    transaction_features: dict = None,
    actual_label: int = None,
):
    """Insert a new prediction record into PostgreSQL."""
    insert_query = """
    INSERT INTO predictions (
        transaction_time,
        amount,
        prediction,
        label,
        fraud_probability,
        model_version,
        created_at,
        transaction_features,
        actual_label,
        labeled_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    RETURNING id;
    """
    if created_at is None:
        created_at = datetime.utcnow()

    features_json = json.dumps(transaction_features) if transaction_features else None
    labeled_at    = datetime.utcnow() if actual_label is not None else None

    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                insert_query,
                (
                    transaction_time,
                    amount,
                    prediction,
                    label,
                    fraud_probability,
                    model_version,
                    created_at,
                    features_json,
                    actual_label,
                    labeled_at,
                ),
            )
            prediction_id = cursor.fetchone()[0]
        conn.commit()
        return prediction_id
    except Exception as e:
        print(f"Error inserting prediction into database: {e}")
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()


def fetch_production_features(limit: int = 1000) -> list:
    """
    Fetch recent production transactions from PostgreSQL for drift detection.
    Returns a list of dicts containing the full transaction features.
    """
    query = """
    SELECT transaction_features
    FROM   predictions
    WHERE  transaction_features IS NOT NULL
    ORDER  BY created_at DESC
    LIMIT  %s;
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
        return [
            row["transaction_features"] if isinstance(row["transaction_features"], dict)
            else json.loads(row["transaction_features"])
            for row in rows
        ]
    except Exception as e:
        print(f"Error fetching production features: {e}")
        return []
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    init_db()
    migrate_db()
