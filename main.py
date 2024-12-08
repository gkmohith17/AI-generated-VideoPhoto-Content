from pathlib import Path
import os
import datetime
import logging
from typing import List, Optional
from pydantic import BaseModel
import sqlite3
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncio
import aiofiles
from databases import Database
import openai
import requests
from dotenv import load_dotenv
from fastapi.responses import HTMLResponse, FileResponse
from diffusers import StableDiffusionPipeline
import torch
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Request

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Database URL
DATABASE_URL = "sqlite:///./content_generation.db"
database = Database(DATABASE_URL)

stable_diffusion_pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
stable_diffusion_pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

# Models
class User(BaseModel):
    user_id: str
    email: str
    notification_time: Optional[str]

class GenerationRequest(BaseModel):
    prompt: str
    user_id: str
    notification_time: Optional[str]

class ContentStatus(BaseModel):
    user_id: str
    prompt: str
    status: str
    video_paths: List[str]
    image_paths: List[str]
    generated_at: datetime.datetime

# Database initialization
async def init_db():
    async with database.connection() as connection:
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                email TEXT NOT NULL,
                notification_time TEXT
            )
        """)
        
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS content_generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                status TEXT NOT NULL,
                video_paths TEXT,
                image_paths TEXT,
                generated_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        """)
        
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS user_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                action TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        """)

@app.on_event("startup")
async def startup():
    await database.connect()
    await init_db()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

async def generate_images(prompt: str, count: int = 5) -> List[str]:
    """Generate images using Stable Diffusion"""
    try:
        image_paths = []
        for i in range(count):
            # Generate the image
            image = stable_diffusion_pipeline(prompt).images[0]
            
            # Save the image locally
            image_path = f"generated_content/images/{datetime.datetime.now().timestamp()}_{i}.png"
            Path(image_path).parent.mkdir(parents=True, exist_ok=True)
            image.save(image_path)
            
            image_paths.append(image_path)
        
        return image_paths
    except Exception as e:
        logger.error(f"Error generating images: {str(e)}")
        raise HTTPException(status_code=500, detail="Image generation failed")
async def generate_videos(prompt: str, count: int = 5) -> List[str]:
    """Generate videos using RunwayML API (placeholder implementation)"""
    # Note: This is a placeholder. You would need to implement actual video generation
    # using your preferred video generation API
    video_paths = []
    for i in range(count):
        # Placeholder path - in reality, this would be the path where the generated video is saved
        video_path = f"generated_content/videos/{datetime.datetime.now().timestamp()}_{i}.mp4"
        video_paths.append(video_path)
    return video_paths

# API Endpoints
@app.post("/generate/")
async def generate_content(request: GenerationRequest):
    """Endpoint to initiate content generation"""
    try:
        # Create directory structure if it doesn't exist
        Path(f"generated_content/{request.user_id}").mkdir(parents=True, exist_ok=True)
        
        # Log the generation request
        query = """
            INSERT INTO content_generations (user_id, prompt, status, generated_at)
            VALUES (:user_id, :prompt, :status, :generated_at)
        """
        values = {
            "user_id": request.user_id,
            "prompt": request.prompt,
            "status": "Processing",
            "generated_at": datetime.datetime.now()
        }
        
        await database.execute(query=query, values=values)
        
        # Start async generation
        asyncio.create_task(process_generation(request))
        
        return {"message": "Content generation started", "status": "Processing"}
    
    except Exception as e:
        logger.error(f"Error initiating content generation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start content generation")

async def process_generation(request: GenerationRequest):
    """Process the content generation asynchronously"""
    try:
        # Generate images and videos concurrently
        image_task = generate_images(request.prompt, count=5)
        video_task = generate_videos(request.prompt, count=5)

        # Wait for both tasks to complete
        image_paths, video_paths = await asyncio.gather(image_task, video_task)

        # Update database with results
        query = """
            UPDATE content_generations
            SET status = :status,
                video_paths = :video_paths,
                image_paths = :image_paths
            WHERE user_id = :user_id
            AND prompt = :prompt
        """
        values = {
            "status": "Completed",
            "video_paths": ",".join(video_paths),
            "image_paths": ",".join(image_paths),
            "user_id": request.user_id,
            "prompt": request.prompt,
        }

        await database.execute(query=query, values=values)

        # Send notification
        await send_notification(request.user_id, request.notification_time)

    except Exception as e:
        logger.error(f"Error processing generation: {str(e)}")
        # Update database with error status
        await database.execute(
            """
            UPDATE content_generations
            SET status = 'Error'
            WHERE user_id = :user_id AND prompt = :prompt
            """,
            values={"user_id": request.user_id, "prompt": request.prompt},
        )

async def send_notification(user_id: str, notification_time: Optional[str]):
    """Send notification to user"""
    # Implement your notification logic here (email, console, etc.)
    logger.info(f"Notification sent to user {user_id}")

# User Management Endpoints
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/users/")
async def create_user(user: User):
    """Create a new user"""
    query = """
        INSERT INTO users (user_id, email, notification_time)
        VALUES (:user_id, :email, :notification_time)
    """
    await database.execute(query=query, values=dict(user))
    return {"message": "User created successfully"}

@app.get("/content/{user_id}")
async def get_user_content(user_id: str):
    """Get user's generated content"""
    query = """
        SELECT * FROM content_generations
        WHERE user_id = :user_id
        ORDER BY generated_at DESC
    """
    result = await database.fetch_all(query=query, values={"user_id": user_id})
    return result

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/generated_content/{file_path:path}")
async def serve_static(file_path: str):
    return FileResponse(f"generated_content/{file_path}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

