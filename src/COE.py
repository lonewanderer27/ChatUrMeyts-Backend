import logging
from PIL import Image
import fitz
import io

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class COE:
    def __init__(self, file_data):
        """
        Initialize the COE object.

        Parameters:
            file_data (bytes or BytesIO): In-memory PDF or image file data.
        """
        self.file_data = file_data
        self.image = None  # Holds the loaded image
        self.target_width = 850  # Target width for resizing
        self.target_height = 1000  # Target height for resizing

        # Predefined coordinates for cropping various sections of the COE
        self.top_image_x = 0
        self.top_image_y = 112
        self.top_image_width = None  # Set dynamically after loading image
        self.top_image_height = 60
        self.semester_x = 300
        self.semester_width = 250
        self.semester_height = 17
        self.student_name_x = 105
        self.student_name_y = 20
        self.student_name_width = 400
        self.student_name_height = 20
        self.course_x = 105
        self.course_y = 40
        self.course_width = 400
        self.course_height = 20
        self.student_no_x = 665
        self.student_no_y = 20
        self.student_no_width = 150
        self.student_no_height = 20
        self.acad_year_x = 665
        self.acad_year_y = 40
        self.acad_year_width = 150
        self.acad_year_height = 20
        self.block_no_x = 275
        self.block_no_y = 0
        self.block_no_width = 100
        self.block_no_height = 25
        self.bottom_image_x = 0
        self.bottom_image_y = 206
        self.cropped_width = 520
        self.cropped_height = None  # Dynamically computed after loading the image

    def load_file(self):
        """
        Load the file (PDF or image). If it's a PDF, extract the first image.
        """
        # Attempt to open as an image
        try:
            self.image = Image.open(io.BytesIO(self.file_data))
            logger.info("Image loaded successfully.")
        except IOError:
            # If not an image, try to open as PDF
            try:
                self.image = self._extract_image_from_pdf()
                logger.info("Image extracted from PDF successfully.")
            except Exception as e:
                logger.error(f"Failed to load file: {e}")
                raise ValueError("Unsupported file format or corrupt data.")

    def _extract_image_from_pdf(self):
        """
        Extract the first image from the PDF.

        Returns:
            PIL Image: The extracted image from the PDF.
        """
        logger.info("Extracting image from PDF...")
        pdf_document = fitz.open(stream=self.file_data, filetype="pdf")

        # Extract the first image from the first page
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            image_list = page.get_images(full=True)
            if image_list:
                for img in image_list:
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    logger.info(f"Image extracted from page {page_num + 1}.")
                    return image
        logger.error("No images found in the PDF.")
        raise ValueError("The provided PDF does not contain any images.")

    def resize_image(self):
        """
        Resize the image to target dimensions (850x1000).
        """
        if not self.image:
            raise ValueError("No image loaded. Please load a file first.")
        self.image = self.image.resize((self.target_width, self.target_height))
        logger.info(f"Image resized to {self.target_width}x{self.target_height}.")

    def get_coe_image(self):
        """
        Get the COE image.

        Returns:
            PIL Image: The COE image.
        """
        if not self.image:
            raise ValueError("No image loaded. Please load a file first.")
        return self.image

    def get_top_image(self):
        """
        Extract and return the top image region based on predefined coordinates.

        Returns:
            PIL Image: The top image.
        """
        if not self.image:
            raise ValueError("No image loaded. Please load a file first.")

        # Dynamically set top image width if not already set
        if not self.top_image_width:
            self.top_image_width = self.image.width

        # Coordinates for cropping the top image
        box = (
            self.top_image_x,
            self.top_image_y,
            self.top_image_x + self.top_image_width,
            self.top_image_y + self.top_image_height,
        )
        top_image = self.image.crop(box)
        logger.info("Top image extracted successfully.")
        return top_image

    def get_bottom_image(self):
        """
        Extract and return the bottom image region based on computed dimensions.

        Returns:
            PIL Image: The bottom image.
        """
        if not self.image:
            raise ValueError("No image loaded. Please load a file first.")

        # Dynamically set cropped height if not already set
        if self.cropped_height is None:
            self.cropped_height = self.target_height - 306

        # Coordinates for cropping the bottom image
        box = (
            self.bottom_image_x,
            self.bottom_image_y,
            self.bottom_image_x + self.cropped_width,
            self.bottom_image_y + self.cropped_height,
        )
        bottom_image = self.image.crop(box)
        logger.info("Bottom image extracted successfully.")
        return bottom_image

    def extract_semester(self):
        """
        Extract the semester image from the top image.

        Returns:
            PIL Image: The cropped semester image.
        """
        return self._extract_text_region("semester")

    def extract_block_no(self):
        """
        Extract and return the block number region from the image.

        Returns:
            PIL Image: The block number image.
        """
        return self._extract_text_region("block_no", from_bottom=True)

    def extract_student_name(self):
        """
        Extract the student name from the top image.

        Returns:
            PIL Image: The student name image.
        """
        return self._extract_text_region("student_name")

    def extract_course(self):
        """
        Extract the course from the top image.

        Returns:
            PIL Image: The course image.
        """
        return self._extract_text_region("course")

    def extract_student_no(self):
        """
        Extract the student number from the top image.

        Returns:
            PIL Image: The student number image.
        """
        return self._extract_text_region("student_no")

    def extract_acad_year(self):
        """
        Extract the academic year from the top image.

        Returns:
            PIL Image: The academic year image.
        """
        return self._extract_text_region("acad_year")

    def _extract_text_region(self, field, from_bottom=False):
        """
        Helper method to extract various fields based on predefined coordinates.

        Parameters:
            field (str): The field to extract.
            from_bottom (bool): Whether to extract from the bottom image.
        
        Returns:
            PIL Image: The extracted region.
        """
        coordinates = {
            "student_name": (
                self.student_name_x,
                self.student_name_y,
                self.student_name_width,
                self.student_name_height,
            ),
            "course": (
                self.course_x,
                self.course_y,
                self.course_width,
                self.course_height,
            ),
            "student_no": (
                self.student_no_x,
                self.student_no_y,
                self.student_no_width,
                self.student_no_height,
            ),
            "acad_year": (
                self.acad_year_x,
                self.acad_year_y,
                self.acad_year_width,
                self.acad_year_height,
            ),
            "semester": (
                self.semester_x,
                0,
                self.semester_width,
                self.semester_height,
            ),
            "block_no": (
                self.block_no_x,
                self.block_no_y,
                self.block_no_width,
                self.block_no_height,
            ),
        }

        if field not in coordinates:
            raise ValueError("Invalid field name.")

        x, y, width, height = coordinates[field]
        if from_bottom:
            image_source = self.get_bottom_image()
        else:
            image_source = self.get_top_image()

        # Crop the text region using predefined dimensions
        field_image = image_source.crop((x, y, x + width, y + height))
        logger.info(f"{field} image extracted successfully.")
        return field_image

    def extract_classes(self):
        """
        Extract and store each class image (class code, unit count, subject name, and schedule) into a list of dictionaries.

        Returns:
            list of dict: List containing class data with images.
        """
        if not self.image:
            raise ValueError("No image loaded. Please load a file first.")

        bottom_image = self.get_bottom_image()

        # Start cropping from the top of the bottom image
        y = 30
        class_index = 1
        classes_data = []

        while y + 45 <= bottom_image.height:
            logger.info(f"Extracting class {class_index}...")

            # Crop the entire class row
            class_image = bottom_image.crop((0, y, self.cropped_width, y + 45))

            # Crop class code (left part, 90px wide)
            class_code_image = class_image.crop((0, 0, 90, 45))

            # Crop unit count (right part, 50px wide)
            unit_count_image = class_image.crop((490, 5, 520, 40))

            # Crop middle part of the class (excluding class code and unit count)
            middle_image = class_image.crop((90, 0, 470, 45))

            # Split the middle part into subject name (top 22px) and schedule (bottom 23px)
            subject_name_image = middle_image.crop((0, 0, 380, 22))
            schedule_image = middle_image.crop((0, 22, 380, 45))

            # Store the class data as a dictionary and append it to the list
            class_data = {
                "class_code": class_code_image,
                "unit_count": unit_count_image,
                "subject_name": subject_name_image,
                "schedule": schedule_image,
            }
            classes_data.append(class_data)

            # Increment y to move to the next class row
            y += 45
            class_index += 1

        logger.info("Class extraction completed.")
        return classes_data