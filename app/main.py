import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import auth, chatbot, counseling, knowledge, messages, prescription, reminders
from app.services.auth_service import initialize_auth_database

load_dotenv()

allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001,https://pharmaassist-ai.vercel.app",
).split(",")
allowed_origin_regex = os.getenv(
    "ALLOWED_ORIGIN_REGEX",
    r"https://[a-zA-Z0-9-]+\.vercel\.app",
)

app = FastAPI(
    title="PharmaAssist AI Backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins if origin.strip()],
    allow_origin_regex=allowed_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prescription.router, prefix="/api/v1")
app.include_router(counseling.router, prefix="/api/v1")
app.include_router(chatbot.router, prefix="/api/v1")
app.include_router(reminders.router, prefix="/api/v1")
app.include_router(auth.router, prefix="/api/v1")
app.include_router(messages.router, prefix="/api/v1")
app.include_router(knowledge.router, prefix="/api/v1")


@app.get("/")
def home():
    return {
        "message": "PharmaAssist AI Backend Running",
        "status": "healthy",
    }


@app.on_event("startup")
async def startup_event():
    initialize_auth_database()
    missing = [
        name
        for name in ["GEMINI_API_KEY"]
        if not os.getenv(name)
    ]
    if missing:
        print(
            "Warning: missing environment variables for AI services: "
            + ", ".join(missing)
        )
    print("🚀 PharmaAssist AI Backend Started")


@app.on_event("shutdown")
async def shutdown_event():
    print("🛑 PharmaAssist AI Backend Stopped")