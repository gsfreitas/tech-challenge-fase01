from dotenv import load_dotenv
import os

# Carrega variáveis do .env
load_dotenv()

# =========================
# AMBIENTE
# =========================
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# =========================
# JWT
# =========================
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-key-change-this")
TOKEN_EXPIRE_MINUTES = int(os.getenv("TOKEN_EXPIRE_MINUTES", 30))

# =========================
# USERS
# =========================
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
USER_PASSWORD = os.getenv("USER_PASSWORD", "user123")


# =========================
# RATE LIMITING
# =========================
REQUESTS_LIMIT = int(os.getenv("REQUESTS_LIMIT", 100))
WINDOW_SECONDS = int(os.getenv("WINDOW_SECONDS", 3600))