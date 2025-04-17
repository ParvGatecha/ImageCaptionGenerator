from pydantic import BaseModel

class CaptionResponse(BaseModel):
    """Response schema for the generated caption."""
    filename: str
    caption: str
    error: str | None = None

class HealthCheckResponse(BaseModel):
    """Response schema for health check."""
    status: str