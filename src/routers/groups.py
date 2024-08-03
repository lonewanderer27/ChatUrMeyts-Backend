from fastapi import APIRouter

router = APIRouter(
    prefix="/groups",
    tags=["Hello"],
)


@router.get("/pid/{profile_id}", description="Get group recommendations")
async def get_group_recommendations(profile_id: str):
    return {"message": "Hello World"}
