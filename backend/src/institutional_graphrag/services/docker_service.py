import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

from institutional_graphrag.storage.database import create_tables

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[3]  # backend
DOCKER_COMPOSE_FILE = BASE_DIR / "docker-compose.yml"


@contextmanager
def manage_docker_services():
    print("Starting Docker services...")

    result = subprocess.run(
        ["docker", "compose", "-f", str(DOCKER_COMPOSE_FILE), "up", "-d"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("Docker compose error:")
        print(result.stderr)
        raise RuntimeError("Docker compose failed")

    db_host = os.getenv("HOST", "localhost")
    db_port = int(os.getenv("POSTGRES_PORT", 5432))
    db_user = os.getenv("POSTGRES_USER", "admin")
    db_password = os.getenv("POSTGRES_PASSWORD", "admin123")
    db_name = os.getenv("POSTGRES_DB", "app_db")

    print("Waiting for Postgres to be ready...")

    ready = False
    for _ in range(15):
        try:
            conn = psycopg2.connect(
                host=db_host,
                port=db_port,
                user=db_user,
                password=db_password,
                database=db_name,
            )
            conn.close()
            ready = True
            break
        except psycopg2.OperationalError:
            print("Postgres not ready yet...")
            time.sleep(2)

    if not ready:
        raise RuntimeError("Postgres did not start in time")

    create_tables()

    try:
        yield
    finally:
        print("Stopping Docker services...")

        subprocess.run(["docker", "compose", "-f", str(DOCKER_COMPOSE_FILE), "down"])
