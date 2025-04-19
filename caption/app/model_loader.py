import logging
from io import BytesIO
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch # Or tensorflow as tf

logger = logging.getLogger(__name__)

# --- Configuration ---
# Choose a pre-trained model from Hugging Face Hub
# Example: "nlpconnect/vit-gpt2-image-captioning"
MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# If using TF: DEVICE = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

# --- Global Variables ---
model = None
feature_extractor = None
tokenizer = None

# --- Initialization ---
def load_model():
    """Loads the pre-trained image captioning model, processor, and tokenizer."""
    global model, feature_extractor, tokenizer
    try:
        logger.info(f"Loading model '{MODEL_NAME}' onto device '{DEVICE}'...")
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(DEVICE)
        feature_extractor = ViTImageProcessor.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        # Depending on requirements, you might want to raise the exception
        # or handle it gracefully (e.g., disable the captioning endpoint)
        raise RuntimeError(f"Failed to load ML model: {e}")

# --- Inference ---
def generate_caption(image_bytes: bytes) -> str:
    """Generates a caption for the given image bytes."""
    global model, feature_extractor, tokenizer

    if not all([model, feature_extractor, tokenizer]):
        raise RuntimeError("Model is not loaded. Cannot generate caption.")

    try:
        logger.debug("Opening image from bytes.")
        img = Image.open(BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert(mode="RGB")
        logger.debug("Image opened successfully.")

        # --- PyTorch Inference ---
        logger.debug("Processing image and generating features...")
        pixel_values = feature_extractor(images=[img], return_tensors="pt").pixel_values.to(DEVICE)
        logger.debug("Generating caption...")
        # Adjust generation parameters as needed (max_length, num_beams, etc.)
        output_ids = model.generate(pixel_values, max_length=32, num_beams=4) # Example parameters
        logger.debug("Decoding caption...")
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        logger.info(f"Generated caption: {caption}")
        return caption

        # --- TensorFlow Inference (Alternative) ---
        # logger.debug("Processing image and generating features...")
        # pixel_values = feature_extractor(images=[img], return_tensors="tf").pixel_values
        # logger.debug("Generating caption...")
        # with tf.device(DEVICE):
        #     output_ids = model.generate(pixel_values, max_length=32, num_beams=4) # Example parameters
        # logger.debug("Decoding caption...")
        # caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        # logger.info(f"Generated caption: {caption}")
        # return caption

    except Exception as e:
        logger.error(f"Error during caption generation: {e}", exc_info=True)
        # Re-raise or return an error indicator
        raise ValueError(f"Caption generation failed: {e}")

# --- Call load_model on application startup (handled in main.py) ---