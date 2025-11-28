# Flowchart AI

## Features

-   **Text-to-Flowchart Conversion**: Automatically generates Mermaid flowcharts from natural language.
-   **Responsive Design**: A mobile-first interface that works on any device.
-   **Real-Time Rendering**: Instantly view your generated flowchart.
-   **FastAPI Backend**: Backend built with Python and FastAPI.

## Tech Stack

-   **Backend**: Python, FastAPI
-   **Frontend**: HTML, CSS, JavaScript
-   **Diagrams**: Mermaid.js

## Getting Started

### Prerequisites

-   Python 3.8+
-   An active internet connection

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Rfannn/flowchart-ai.git
    cd flowchart-ai
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Start the server:**
    ```bash
    uvicorn main:app --reload
    ```

2.  **Open your browser:**
    Navigate to `http://127.0.0.1:8000` to use the application.

## Usage

1.  Enter a description of your process in the text area.
2.  Click the "Generate Diagram" button.
3.  The flowchart will be rendered in the container below.

## API Endpoints

Flowchart AI exposes a simple REST API for generating flowcharts.

-   `POST /generate`: Converts a text description to a Mermaid flowchart.
    -   **Request Body**: `{"text": "Your process description."}`
    -   **Response**: A JSON object with the flowchart steps, decisions, and Mermaid code.

-   `GET /health`: Returns the health status of the application.
