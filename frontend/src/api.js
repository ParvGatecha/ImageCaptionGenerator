// api.js

// Base URLs for the different services
const CAPTION_API_BASE_URL =
  process.env.REACT_APP_CAPTION_API_URL || "http://ai-tools/api";
const DETECTION_API_BASE_URL =
  process.env.REACT_APP_DETECTION_API_URL || "http://ai-tools/api";

// Helper function to handle common fetch logic and errors
const fetchApi = async (url, options, file) => {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch(url, {
      method: "POST",
      body: formData,
      ...options, // Allow passing other options if needed
    });

    if (!response.ok) {
      let errorDetail = `HTTP error! status: ${response.status}`;
      try {
        const errorData = await response.json();
        // Prioritize the custom 'error' field if present
        errorDetail = errorData.error || errorData.detail || errorDetail;
      } catch (jsonError) {
        console.warn("Could not parse error response as JSON", jsonError);
        // Try reading as text for non-JSON errors
        try {
          const textError = await response.text();
          errorDetail = `${errorDetail}: ${textError}`;
        } catch (textParseError) {
          // Keep the original HTTP error status if text parsing fails
        }
      }
      throw new Error(errorDetail);
    }

    return await response.json(); // Return the parsed JSON data
  } catch (error) {
    console.error(`Error during API call to ${url}:`, error);
    // Re-throw the error with a potentially more specific message
    throw new Error(error.message || "Network or API error occurred.");
  }
};

// --- Captioning API ---
export const uploadImageAndGetCaption = async (file) => {
  const data = await fetchApi(`${CAPTION_API_BASE_URL}/caption`, {}, file);
  if (data.error) {
    throw new Error(data.error); // Throw error if backend indicated one specifically
  }
  if (!data.caption && !data.error) {
    // Handle cases where caption might be empty but no explicit error
    console.warn("Received response without caption or error field.");
    return ""; // Return empty string or handle as needed
  }
  return data.caption; // Return the caption string
};

// --- Object Detection API ---
export const uploadImageAndDetectObjects = async (file) => {
  const data = await fetchApi(`${DETECTION_API_BASE_URL}/object`, {}, file);
  console.log("Raw API Response:", data); // <-- Add this log
  if (data.error) {
    throw new Error(data.error);
  }
  if (!data.objects && !data.error) {
    console.warn("Received response without objects or error field.");
    return [];
  }
  console.log("Returning Objects:", data.objects); // <-- Add this log
  return data.objects || [];
};

// Optional: Add health check functions if needed
export const checkCaptionHealth = async () => {
  try {
    const response = await fetch(`${CAPTION_API_BASE_URL}/health`);
    if (!response.ok)
      throw new Error(`Caption service unhealthy: ${response.status}`);
    return await response.json();
  } catch (error) {
    console.error("Caption health check failed:", error);
    return { status: "unreachable" };
  }
};

export const checkDetectionHealth = async () => {
  try {
    const response = await fetch(`${DETECTION_API_BASE_URL}/health`);
    if (!response.ok)
      throw new Error(`Detection service unhealthy: ${response.status}`);
    return await response.json();
  } catch (error) {
    console.error("Detection health check failed:", error);
    return { status: "unreachable" };
  }
};
