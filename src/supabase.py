from supabase import create_client, Client
from typing import List
import logging
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# Initialize the Supabase client synchronously
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_all_student_profiles():
    """
    Fetch all student profiles using the RPC function.
    """
    logger.info("Calling RPC: get_all_student_profiles")
    response = supabase.rpc("get_all_student_profiles").execute()
    logger.debug(f"get_all_student_profiles response: {response}")
    
    # Safely access the 'error' attribute
    error = getattr(response, 'error', None)
    if error:
        logger.error(f"Error fetching student profiles: {error.message}")
        raise Exception(f"Error fetching student profiles: {error.message}")
    
    logger.info("Successfully fetched student profiles.")
    return response.data

def get_group_members():
    """
    Fetch all group members using the RPC function.
    """
    logger.info("Calling RPC: get_group_members")
    response = supabase.rpc("get_group_members").execute()
    logger.debug(f"get_group_members response: {response}")
    
    # Safely access the 'error' attribute
    error = getattr(response, 'error', None)
    if error:
        logger.error(f"Error fetching group members: {error.message}")
        raise Exception(f"Error fetching group members: {error.message}")
    
    logger.info("Successfully fetched group members.")
    return response.data

def get_group_metadata():
    """
    Fetch all group metadata using the RPC function.
    """
    logger.info("Calling RPC: get_group_metadata")
    response = supabase.rpc("get_group_metadata").execute()
    logger.debug(f"get_group_metadata response: {response}")
    
    # Safely access the 'error' attribute
    error = getattr(response, 'error', None)
    if error:
        logger.error(f"Error fetching group metadata: {error.message}")
        raise Exception(f"Error fetching group metadata: {error.message}")
    
    logger.info("Successfully fetched group metadata.")
    return response.data

def save_recommendation(student_id: int, group_ids: List[int]):
    """
    Save the group id recommendations for a student
    """

    logger.info(f"Saving recommendation for student_id: {student_id}")
    logger.info(f"Group IDs: {group_ids}")
    response = (
        supabase.table("student_recommend_groups")
        .insert({"student_id": student_id, "group_ids": group_ids})
        .execute()
    )

    # Safely access the 'error' attribute
    error = getattr(response, 'error', None)
    if error:
        logger.error(f"Error saving recommendation: {error.message}")
        raise Exception(f"Error saving recommendation: {error.message}")
    
    logger.info("Successfully saved recommendation.")
    return response.data