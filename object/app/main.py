import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator

from . import model_loader, schemas # Use relative imports within the package

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Lifespan Management (for loading model on startup) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    logger.info("Application startup: Loading Object Detection model...")
    try:
        model_loader.load_model()
        logger.info("Object Detection Model loaded successfully during startup.")
    except Exception as e:
        logger.error(f"Fatal Error: Failed to load Object Detection model during startup: {e}", exc_info=True)
        # Allow app to start, but endpoints needing the model will fail.
    yield
    # Clean up resources if needed
    logger.info("Application shutdown: Cleaning up resources...")


# --- FastAPI App Initialization ---
API_PREFIX = "/api"

app = FastAPI(
    title="Object Detection API",
    description="API to detect objects in uploaded images.",
    openapi_url=f"{API_PREFIX}/openapi.json",
    openapi_prefix=API_PREFIX,
    lifespan=lifespan
)

# Configure Prometheus metrics with more detailed instrumentation
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="fastapi_inprogress",
    inprogress_labels=True,
)

# Add more detailed metrics
instrumentator.add(
    prometheus_multiproc_dir=".",
    default_labels={
        "app": "object-detection-service",
        "version": "1.0.0"
    }
)

# Instrument the app and expose metrics at root path
instrumentator.instrument(app).expose(app, include_in_schema=True, should_gzip=True)

# Add a metrics endpoint explicitly
@app.get("/metrics", include_in_schema=False)
async def metrics():
    return instrumentator.metrics()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# --- API Endpoints ---
@app.get("/api/health", response_model=schemas.HealthCheckResponse, tags=["Health"])
async def health_check():
    """Performs a basic health check, including model status."""
    # Check only for the model, as image_processor is removed
    model_ready = bool(model_loader.model)
    status_msg = "ready" if model_ready else "model_loading_failed_or_in_progress"
    logger.info(f"Health check requested. Model status: {status_msg}")
    return schemas.HealthCheckResponse(status=status_msg)

@app.post("/api/object", response_model=schemas.ObjectDetectionResponse, tags=["Detection"])
async def detect_objects_endpoint(file: UploadFile = File(...)):
    """
    Uploads an image file and returns detected objects with bounding boxes.

    - **file**: The image file to upload (e.g., JPEG, PNG).
    """
    logger.info(f"Received request for object detection: {file.filename}")
    if not file.content_type or not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type received: {file.content_type}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Please upload an image.",
        )

    # Ensure model is ready before processing
    # Ensure model is ready before processing
    if not model_loader.model: # <-- Corrected line (only check for model)
        logger.error("Detection request failed: Model is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready or failed to load. Please try again later.",
        )

    try:
        image_bytes = await file.read()
        logger.debug(f"Read {len(image_bytes)} bytes from uploaded file: {file.filename}")

        # Perform detection
        detected_objects_data = model_loader.detect_objects(image_bytes)

        # Format response using Pydantic models
        detected_objects = [schemas.DetectedObject(**obj_data) for obj_data in detected_objects_data]

        logger.info(f"Successfully processed object detection for {file.filename}")
        return schemas.ObjectDetectionResponse(
            filename=file.filename,
            objects=detected_objects
        )

    except ValueError as ve: # Specific error from our detection function
         logger.error(f"Value error during detection for {file.filename}: {ve}", exc_info=True)
         # Return response with error message
         return schemas.ObjectDetectionResponse(filename=file.filename, objects=[], error=f"Detection error: {ve}")
    except HTTPException as http_exc: # Re-raise client/server errors FastAPI knows about
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error during detection for {file.filename}: {e}", exc_info=True)
        # Return a response with a generic error message for unexpected issues
        return schemas.ObjectDetectionResponse(
            filename=file.filename,
            objects=[],
            error="An unexpected error occurred during object detection."
        )
    finally:
         # Ensure the file handle is closed even if errors occur
         await file.close()
         logger.debug(f"Closed file handle for: {file.filename}")

# To run locally (for development): uvicorn object_detection.app.main:app --reload --port 8001
# Note: Use a different port (e.g., 8001) if caption service runs on 8000