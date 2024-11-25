import os
import uvicorn
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routers.hello import router as hello_router
from src.routers.groups import router as groups_router
from src.routers.students import router as students_router
from src.routers.image import router as image_router

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

chaturmates = FastAPI(
    title="Chat-Ur-Meyts API",
    description="API for Chat-Ur-Meyts",
    version="1.0"
)

chaturmates.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["http://localhost:5173", "https://chat-ur-meyts.vercel.app", "https://chaturmates-v2.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"]
)

chaturmates.include_router(groups_router)
chaturmates.include_router(image_router)
chaturmates.include_router(students_router)
chaturmates.include_router(hello_router)

@chaturmates.get("/", tags=["Root"])
async def root():
    return {"message": "Hello Klasmeyt!"}

if __name__ == "__main__":
    uvicorn.run(chaturmates, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))