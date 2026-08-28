from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import prescription
from app.routes import counseling
from app.routes import chatbot
from app.routes import reminders


app = FastAPI(
    title="PharmaAssist AI Backend",
    version="1.0.0"
)


# Allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routes
app.include_router(
    prescription.router,
    prefix="/api/v1"
)

app.include_router(
    counseling.router,
    prefix="/api/v1"
)

app.include_router(
    chatbot.router,
    prefix="/api/v1"
)

app.include_router(
    reminders.router,
    prefix="/api/v1"
)


@app.get("/")
def home():
    return {
        "message": "PharmaAssist AI Backend Running",
        "status": "healthy"
    }


@app.on_event("startup")
async def startup_event():
    print("🚀 PharmaAssist AI Backend Started")


@app.on_event("shutdown")
async def shutdown_event():
    print("🛑 PharmaAssist AI Backend Stopped")