from fastapi import APIRouter

router = APIRouter(prefix="/hello", tags=["Root"])


@router.get("/", description="Say hello to the world")
async def say_hello():
    return {"message": "Hello World"}


@router.get("/{name}", description="Say hello to a klasmeyt")
async def say_hello(name: str):
    return {"message": f"Hello klasmeyt {name}"}
