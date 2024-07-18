from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routers.hello import router as hello_router
import os
import uvicorn

chaturmates = FastAPI(
    title="Chaturmates API",
    description="API for Chaturmates",
    version="0.1.0"
)

chaturmates.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

chaturmates.include_router(hello_router)


@chaturmates.get("/")
async def root():
    return {"message": "Hello Klasmeyt!"}


if __name__ == "__main__":
    uvicorn.run(chaturmates, host="0.0.0.0", port=int(
        os.environ.get("PORT", 2428)))
