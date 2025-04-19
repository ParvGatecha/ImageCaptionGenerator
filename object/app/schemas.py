from pydantic import BaseModel, Field
from typing import List, Tuple

class DetectedObject(BaseModel):
    """Schema for a single detected object."""
    label: str = Field(..., description="Class label of the detected object.")
    score: float = Field(..., description="Confidence score of the detection.", ge=0.0, le=1.0)
    # Box format: [xmin, ymin, xmax, ymax] relative to image dimensions
    box: List[int] = Field(..., description="Bounding box coordinates [xmin, ymin, xmax, ymax].")

class ObjectDetectionResponse(BaseModel):
    """Response schema for the object detection endpoint."""
    filename: str
    objects: List[DetectedObject]
    error: str | None = None

class HealthCheckResponse(BaseModel):
    """Response schema for health check."""
    status: str