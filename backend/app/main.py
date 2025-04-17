import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from . import model_loader, schemas # Use relative imports within the package

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Lifespan Management (for loading model on startup) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    logger.info("Application startup: Loading ML model...")
    try:
        model_loader.load_model()
        logger.info("ML Model loaded successfully during startup.")
    except Exception as e:
        logger.error(f"Fatal Error: Failed to load ML model during startup: {e}", exc_info=True)
        # Depending on policy, you might want the app to fail startup here
        # For now, we log the error and continue; endpoints needing the model will fail.
    yield
    # Clean up the ML models and release the resources
    logger.info("Application shutdown: Cleaning up resources...")
    # Add any cleanup logic here if needed (e.g., releasing GPU memory explicitly)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Image Captioning API",
    description="API to generate captions for uploaded images.",
    version="0.1.0",
    lifespan=lifespan # Use the lifespan context manager
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # or ["*"] to allow all (less secure)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---
@app.get("/health", response_model=schemas.HealthCheckResponse, tags=["Health"])
async def health_check():
    """Performs a basic health check."""
    # Could add more checks here (e.g., model loaded status)
    model_ready = all([model_loader.model, model_loader.feature_extractor, model_loader.tokenizer])
    status_msg = "ready" if model_ready else "model_loading_failed"
    logger.info(f"Health check requested. Status: {status_msg}")
    return schemas.HealthCheckResponse(status=status_msg)

@app.post("/caption", response_model=schemas.CaptionResponse, tags=["Captioning"])
async def create_caption(file: UploadFile = File(...)):
    """
    Uploads an image file and returns a generated caption.

    - **file**: The image file to upload (e.g., JPEG, PNG).
    """
    logger.info(f"Received request to caption image: {file.filename}")
    if not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type received: {file.content_type}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Please upload an image.",
        )

    # Ensure model is ready before processing
    if not all([model_loader.model, model_loader.feature_extractor, model_loader.tokenizer]):
        logger.error("Caption request failed: Model is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready or failed to load. Please try again later.",
        )

    try:
        image_bytes = await file.read()
        logger.debug(f"Read {len(image_bytes)} bytes from uploaded file.")
        caption = model_loader.generate_caption(image_bytes)
        logger.info(f"Successfully generated caption for {file.filename}")
        return schemas.CaptionResponse(filename=file.filename, caption=caption)
    except ValueError as ve: # Specific error from our generation function
         logger.error(f"Value error during captioning for {file.filename}: {ve}", exc_info=True)
         return schemas.CaptionResponse(filename=file.filename, caption="", error=f"Caption generation error: {ve}")
    except HTTPException as http_exc: # Re-raise HTTP exceptions
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error during captioning for {file.filename}: {e}", exc_info=True)
        # Return a generic error response instead of raising HTTPException
        # to potentially mask internal details, but log the full error.
        # Alternatively, raise HTTPException(500, "Internal server error")
        return schemas.CaptionResponse(filename=file.filename, caption="", error="An unexpected error occurred during caption generation.")
    finally:
         await file.close() # Ensure file handle is closed

# --- Add more endpoints as needed (e.g., model info, batch processing) ---

# To run locally (for development): uvicorn backend.app.main:app --reload --port 8000