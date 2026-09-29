"""
Main FastAPI Application Entrypoint for Smart Edu API Gateway.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router as assessment_router

app = FastAPI(
    title="Smart Edu - Online Assessment REST API Gateway",
    description=(
        "Integration Gateway for Online Assessment Platforms (Azota / Study4 / Tuyensinh247). "
        "Provides asynchronous submission endpoints, CoT misconception diagnosis, "
        "Knowledge Graph mastery propagation, and automated ZPD remediation path generation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(assessment_router)


@app.get("/")
async def root():
    return {
        "message": "Welcome to Smart Edu Assessment REST API Gateway",
        "docs": "/docs",
        "version": "1.0.0"
    }
