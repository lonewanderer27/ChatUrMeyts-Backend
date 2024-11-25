import logging
from PIL import Image, ImageOps
import fitz
import io
import os

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class COE:
    def __init__(
            self,
            file,
            save_images=True,  # Set to False if you don't want to save the extracted images
            save_path="images"  # Folder path where images will be saved. Default is 'images'.
    ):
        """
        Initialize the COE object.

        Parameters:
            file (str): Path to a PDF or image file.
            save_images (bool): Boolean to control if images should be saved. Default is True.
            save_path (str): Folder path where images will be saved. Default is 'images'.
        """
        self.file = file
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

        # Save configuration
        self.save_images = save_images
        self.save_path = save_path

        # Create save directory if it doesn't exist
        if self.save_images and not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def load_file(self):
        """
        Load the file (PDF or image). If it's a PDF, extract the first image.
        """
        if self.file.lower().endswith(".pdf"):
            self.image = self._extract_image_from_pdf()
        elif self.file.lower().endswith((".png", ".jpg", ".jpeg")):
            self.image = Image.open(self.file)
        else:
            raise ValueError("Unsupported file format. Please provide a PDF or image.")
        logger.info("File loaded successfully.")

    def _extract_image_from_pdf(self):
        """
        Extract the first image from the PDF.

        Returns:
            PIL Image: The extracted image from the PDF.
        """
        logger.info("Extracting image from PDF...")
        pdf_document = fitz.open(self.file)

        # Extract the first image from the first page
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            for img in page.get_images(full=True):
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

    def get_coe_image(self, save_image=None):
        """
        Extract and return the COE image.

        Parameters:
            save_image (bool): Boolean to override the class-level save configuration.
        """
        if not self.image:
            raise ValueError("No image loaded. Please load a file first.")

        coe_image = self.image
        logger.info("COE image extracted successfully.")
        if save_image if save_image is not None else self.save_images:
            top_image_path = os.path.join(self.save_path, "coe_image.png")
            top_image.save(top_image_path)
            logger.info(f"COE image saved as {top_image_path}.")

        return coe_image

    def get_top_image(self, save_image=None):
        """
        Extract and return the top image region based on pre-defined coordinates.

        Parameters:
            save_image (bool): Boolean to override the class-level save configuration.
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

        # Save the top image if required
        if save_image if save_image is not None else self.save_images:
            top_image_path = os.path.join(self.save_path, "top_image.png")
            top_image.save(top_image_path)
            logger.info(f"Top image saved as {top_image_path}.")

        return top_image

    def get_bottom_image(self, save_image=None):
        """
        Extract and return the bottom image region based on computed dimensions.

        Parameters:
            save_image (bool): Boolean to override the class-level save configuration.
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

        # Save the bottom image if required
        if save_image if save_image is not None else self.save_images:
            bottom_image_path = os.path.join(self.save_path, "bottom_image.png")
            bottom_image.save(bottom_image_path)
            logger.info(f"Bottom image saved as {bottom_image_path}.")

        return bottom_image

    def extract_semester(self, save_image=None):
        """
        Extract the semester image from the top image using the predefined coordinates and dimensions.

        Returns:
            PIL Image: The cropped semester image.
        """
        if not self.image:
            raise ValueError("No image loaded. Please load a file first.")

        # Fixed coordinates for the semester image
        x = self.semester_x  # Starting x-coordinate from your predefined value
        y = 0  # Starting y-coordinate (you can adjust this if needed)

        # Extract the semester image from the top image
        top_image = self.get_top_image()

        # Crop the semester image using the predefined width and height
        semester_image = top_image.crop((x, y, x + self.semester_width, y + self.semester_height))

        # Optionally, save the image if required
        if save_image if save_image is not None else self.save_images:
            semester_image_path = os.path.join(self.save_path, "semester_image.png")
            semester_image.save(semester_image_path)
            logger.info(f"Semester image saved as {semester_image_path}.")

        # Return the cropped semester image
        return semester_image

    def extract_block_no(self, save_image=None):
        """
        Extract and return the block number region from the image using defined coordinates.

        Parameters:
            save_image: Boolean to override the class-level save configuration.
        """
        if not self.image:
            raise ValueError("No image loaded. Please load a file first.")

        # Define the block number crop area using the class-level coordinates
        block_no_box = (
            self.block_no_x,
            self.block_no_y,
            self.block_no_x + self.block_no_width,
            self.block_no_y + self.block_no_height,
        )

        # Get the bottom image
        bottom_image = self.get_bottom_image()

        # Crop the block number region
        block_no_image = bottom_image.crop(block_no_box)
        logger.info("Block number image extracted successfully.")

        # Decide whether to save based on the provided config
        if save_image if save_image is not None else self.save_images:
            block_no_image_path = os.path.join(self.save_path, "block_no_image.png")
            block_no_image.save(block_no_image_path)
            logger.info(f"Block number image saved as {block_no_image_path}.")

        return block_no_image

    def extract_student_name(self):
        """
        Extract the student name from the top image using the predefined coordinates and dimensions.
        """
        return self._extract_text_region("student_name")

    def extract_course(self):
        """
        Extract the course from the top image using the predefined coordinates and dimensions.
        """
        return self._extract_text_region("course")

    def extract_student_no(self):
        """
        Extract the student number from the top image using the predefined coordinates and dimensions.
        """
        return self._extract_text_region("student_no")

    def extract_acad_year(self):
        """
        Extract the academic year from the top image using the predefined coordinates and dimensions.
        """
        return self._extract_text_region("acad_year")

    def _extract_text_region(self, field):
        """
        Helper method to extract various fields (student_name, course, student_no, acad_year) based on predefined coordinates.
        """
        coordinates = {
            "student_name": (
            self.student_name_x, self.student_name_y, self.student_name_width, self.student_name_height),
            "course": (self.course_x, self.course_y, self.course_width, self.course_height),
            "student_no": (self.student_no_x, self.student_no_y, self.student_no_width, self.student_no_height),
            "acad_year": (self.acad_year_x, self.acad_year_y, self.acad_year_width, self.acad_year_height),
        }

        if field not in coordinates:
            raise ValueError("Invalid field name.")

        x, y, width, height = coordinates[field]
        top_image = self.get_top_image()

        # Crop the text region using predefined dimensions
        field_image = top_image.crop((x, y, x + width, y + height))

        # Save if required
        if self.save_images:
            field_image_path = os.path.join(self.save_path, f"{field}_image.png")
            field_image.save(field_image_path)
            logger.info(f"{field.capitalize()} image saved as {field_image_path}.")

        return field_image

    def extract_classes(self, save_image=None):
        """
        Extract and store each class image (class code, unit count, subject name, and schedule) into a list of dictionaries
        and return it.

        Parameters:
            save_image: Boolean to override the class-level save configuration.
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
            class_code_image_data = class_code_image  # Store the image data here
            if save_image if save_image is not None else self.save_images:
                class_code_image_path = os.path.join(self.save_path, f"class_code_{class_index}.png")
                class_code_image.save(class_code_image_path)
                logger.info(f"Class {class_index} class code saved as {class_code_image_path}.")

            # Crop unit count (right part, 50px wide)
            unit_count_image = class_image.crop((490, 5, 520, 40))
            unit_count_image_data = unit_count_image  # Store the image data here
            if save_image if save_image is not None else self.save_images:
                unit_count_image_path = os.path.join(self.save_path, f"unit_count_{class_index}.png")
                unit_count_image.save(unit_count_image_path)
                logger.info(f"Class {class_index} unit count saved as {unit_count_image_path}.")

            # Crop middle part of the class (excluding class code and unit count)
            middle_image = class_image.crop((90, 0, 470, 45))

            # Split the middle part into subject name (top 22px) and schedule (bottom 23px)
            subject_name_image = middle_image.crop((0, 0, 380, 22))
            subject_name_image_data = subject_name_image  # Store the image data here
            if save_image if save_image is not None else self.save_images:
                subject_name_image_path = os.path.join(self.save_path, f"subject_name_{class_index}.png")
                subject_name_image.save(subject_name_image_path)
                logger.info(f"Class {class_index} subject name saved as {subject_name_image_path}.")

            schedule_image = middle_image.crop((0, 22, 380, 45))
            schedule_image_data = schedule_image  # Store the image data here
            if save_image if save_image is not None else self.save_images:
                schedule_image_path = os.path.join(self.save_path, f"schedule_{class_index}.png")
                schedule_image.save(schedule_image_path)
                logger.info(f"Class {class_index} schedule saved as {schedule_image_path}.")

            # Store the class data as a dictionary and append it to the list
            class_data = {
                "class_code": class_code_image_data,
                "unit_count": unit_count_image_data,
                "subject_name": subject_name_image_data,
                "schedule": schedule_image_data
            }
            classes_data.append(class_data)

            # Increment y to move to the next class row
            y += 45
            class_index += 1

        logger.info("Class extraction completed.")
        return classes_data