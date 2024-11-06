# Chat-Ur-Meyts API

This is the API for Chat-Ur-Meyts, built using FastAPI.

## Requirements

- Python 3.10 or higher
- `pip` (Python package installer)

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/lonewanderer27/ChatUrMeyts-Backend
    cd ChatUrMeyts-Backend
    ```

2. Create and activate a virtual environment:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:

    ```sh
    pip install -r requirements.txt
    ```

## Running the Project

To run the project, use the following command:

```sh
uvicorn src.main:chaturmates --reload