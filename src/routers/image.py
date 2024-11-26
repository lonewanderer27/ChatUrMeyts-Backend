from fastapi import APIRouter, File, UploadFile, HTTPException
import logging
from io import BytesIO
from starlette.responses import StreamingResponse
from ..COE import COE

router = APIRouter(prefix="/image", tags=["Image"])
logger = logging.getLogger(__name__)

async def process_coe_file(coe_file: UploadFile) -> COE:
    """
    Helper function to process the uploaded COE PDF file.

    Parameters:
        coe_file (UploadFile): The uploaded COE PDF file.

    Returns:
        COE: An instance of the COE class after processing.
    """
    if not coe_file.filename or not coe_file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a valid PDF file.")

    try:
        # Read the uploaded file into memory
        file_bytes = await coe_file.read()

        # Initialize COE object with in-memory bytes
        coe_instance = COE(file_bytes)

        # Load and resize the COE image
        coe_instance.load_file()
        coe_instance.resize_image()

        return coe_instance

    except AttributeError as ae:
        logger.error(f"Attribute error while processing {getattr(coe_file, 'filename', 'unknown file')}: {str(ae)}")
        raise HTTPException(status_code=400, detail="File processing failed. Ensure the uploaded file is a valid PDF.")
    except Exception as e:
        logger.error(f"Unexpected error while processing {getattr(coe_file, 'filename', 'unknown file')}: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred. Please try again.")

def image_response(image):
    """
    Convert a PIL image to a StreamingResponse.
    """
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return StreamingResponse(img_byte_arr, media_type="image/png")

class Responses:
    @staticmethod
    def png_image_response(description: str = "A PNG image file."):
        return {
            200: {
                "description": description,
                "content": {"image/png": {}},
            },
            400: {
                "description": "File processing failed. Ensure the uploaded file is a valid PDF.",
            },
            500: {
                "description": "An unexpected error occurred."
            }
        }


@router.post("", description="Extract the image of the COE PDF",
             responses=Responses.png_image_response("A PNG image of the COE PDF."))
async def extract_image_from_pdf(coe: UploadFile = File(...)):
    logger.info("Extracting image from COE PDF")

    coe_instance = await process_coe_file(coe)

    # Extract the COE image
    coe_image = coe_instance.get_coe_image()

    # Return the image as a StreamingResponse
    return image_response(coe_image)

@router.post("/course", description="Extract the course name image of the COE PDF",
             responses=Responses.png_image_response("A PNG image of the course name section."))
async def extract_course_image_from_pdf(coe: UploadFile = File(...)):
    logger.info("Extracting course name image from COE PDF")

    coe_instance = await process_coe_file(coe)

    # Extract course image
    course_image = coe_instance.extract_course()

    # Return the image as a StreamingResponse
    return image_response(course_image)

@router.post("/student_name", description="Extract the student name image of the COE PDF",
             responses=Responses.png_image_response("A PNG image of the student name section."))
async def extract_student_name_image_from_pdf(coe: UploadFile = File(...)):
    logger.info("Extracting student name image from COE PDF")

    coe_instance = await process_coe_file(coe)

    # Extract student name image
    student_name_image = coe_instance.extract_student_name()

    # Return the image as a StreamingResponse
    return image_response(student_name_image)

@router.post("/student_no", description="Extract the student number image of the COE PDF",
             responses=Responses.png_image_response("A PNG image of the student number section."))
async def extract_student_no_image_from_pdf(coe: UploadFile = File(...)):
    logger.info("Extracting student number image from COE PDF")

    coe_instance = await process_coe_file(coe)

    # Extract student number image
    student_no_image = coe_instance.extract_student_no()

    # Return the image as a StreamingResponse
    return image_response(student_no_image)

@router.post("/acad_year", description="Extract the academic year image of the COE PDF",
             responses=Responses.png_image_response("A PNG image of the academic year section."))
async def extract_year_level_image_from_pdf(coe: UploadFile = File(...)):
    logger.info("Extracting acad year image from COE PDF")

    coe_instance = await process_coe_file(coe)

    # Extract acad year image
    acad_year_image = coe_instance.extract_acad_year()

    # Return the image as a StreamingResponse
    return image_response(acad_year_image)

@router.post("/year_level", description="Extract the year level image of the COE PDF",
             responses=Responses.png_image_response("A PNG image of the year level section."))
async def extract_year_level_image_from_pdf(coe: UploadFile = File(...)):
    logger.info("Extracting year level image from COE PDF")

    coe_instance = await process_coe_file(coe)

    # Extract year level image
    year_level_image = coe_instance.extract_acad_year()

    # Return the image as a StreamingResponse
    return image_response(year_level_image)

@router.post("/semester", description="Extract the semester image of the COE PDF",
             responses=Responses.png_image_response("A PNG image of the semester section."))
async def extract_semester_image_from_pdf(coe: UploadFile = File(...)):
    logger.info("Extracting semester image from COE PDF")

    coe_instance = await process_coe_file(coe)

    # Extract semester image
    semester_image = coe_instance.extract_semester()

    # Return the image as a StreamingResponse
    return image_response(semester_image)

@router.post("/block_no", description="Extract the block number image of the COE PDF",
             responses=Responses.png_image_response("A PNG image of the block number section."))
async def extract_top_image_from_pdf(coe: UploadFile = File(...)):
    logger.info("Extracting block number image from COE PDF")

    coe_instance = await process_coe_file(coe)

    # Extract block number image
    block_no_image = coe_instance.extract_block_no()

    # Return the image as a StreamingResponse
    return image_response(block_no_image)

@router.post("/top", description="Extract the top part image of the COE PDF",
             responses=Responses.png_image_response("A PNG image of the top section"))
async def extract_top_image_from_pdf(coe: UploadFile = File(...)):
    logger.info("Extracting top image from COE PDF")

    coe_instance = await process_coe_file(coe)

    # Extract top image
    top_image = coe_instance.get_top_image()

    # Return the image as a StreamingResponse
    return image_response(top_image)

@router.post("/bottom", description="Extract the bottom part image of the COE PDF",
             responses=Responses.png_image_response("A PNG image of the bottom section"))
async def extract_top_image_from_pdf(coe: UploadFile = File(...)):
    logger.info("Extracting bottom image from COE PDF")

    coe_instance = await process_coe_file(coe)

    # Extract bottom image
    bottom_image = coe_instance.get_bottom_image()

    # Return the image as a StreamingResponse
    return image_response(bottom_image)