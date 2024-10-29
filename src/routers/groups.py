import os
from fastapi import APIRouter, HTTPException, Query
from ..supabase import supabase
from pprint import pprint
from typing import List, Any
from pydantic import BaseModel
from ..model import GroupRecommender
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the global recommender to None
recommender = None  # Global variable

router = APIRouter(prefix="/groups")

class Group(BaseModel):
    group_id: int
    group_name: str
    group_subjects: List[str] = []
    year_level: List[int]
    block: str

class GroupIDsResponse(BaseModel):
    student_id: int
    group_ids: List[int] = []
    groups: List[Group] = []

@router.on_event("startup")
async def startup_event():
    global recommender  # Use the global keyword to modify the global variable
    logger.info("Initializing the Group Recommender Model...")
    
    try:
        # Assign a new instance to the global variable
        recommender = GroupRecommender()
        recommender.initialize()
        logger.info("Model initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize the model: {e}")
        raise e  # Prevents the server from starting if initialization fails

@router.get("/sid/{student_id}", 
    response_model=GroupIDsResponse, 
    description="Get group recommendations for a student", 
    name="Get group recommendations for a student")
async def get_group_recommendations_for_student(
    student_id: int,
    top_k: int = Query(10, description="Number of recommendations to return")
):
    try:
        # Use the global recommender instance
        recommendations = recommender.recommend_groups_for_student(student_id, top_k)
        if not recommendations or len(recommendations) == 0:
            raise HTTPException(status_code=404, detail="No recommendations found.")
        # Find the raw group data from the database
        raw_groups = recommender.get_group_chats_by_ids(recommendations)
        
        return GroupIDsResponse(student_id=student_id, group_ids=recommendations, groups=raw_groups)
    except Exception as e:
        logger.error(f"Error during recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
