import React, { useState, useCallback, useEffect, useRef } from "react";
import {
  uploadImageAndGetCaption,
  uploadImageAndDetectObjects,
  // Optional: checkCaptionHealth, checkDetectionHealth
} from "./api"; // Assuming api.js is in the same folder
import styled, {
  createGlobalStyle,
  css,
  keyframes,
  ThemeProvider,
} from "styled-components";
import { motion, AnimatePresence } from "framer-motion";
import {
  FiUpload,
  FiImage,
  FiVolume2,
  FiLoader,
  FiMoon,
  FiSun,
  FiEye, // Icon for detection
  FiMessageSquare, // Icon for caption
  FiXCircle, // Icon for errors
  FiCheckCircle, // Icon for success/ready
} from "react-icons/fi";

// --- Global Styles and Theme ---
const GlobalStyle = createGlobalStyle`
  body {
    background: ${({ theme }) => theme.body};
    color: ${({ theme }) => theme.text};
    font-family: 'Poppins', sans-serif;
    transition: background 0.3s ease-in-out, color 0.3s ease-in-out;
    margin: 0;
    padding: 0;
    line-height: 1.6;
  }

  *, *::before, *::after {
    box-sizing: border-box;
  }
`;

const lightTheme = {
  body: "#f8fafc", // Lighter gray
  text: "#111827", // Dark gray
  cardBg: "#ffffff",
  buttonPrimaryBg: "#2563eb",
  buttonPrimaryHover: "#1e40af",
  buttonSecondaryBg: "#059669",
  buttonSecondaryHover: "#047857",
  accent: "#3b82f6",
  borderColor: "#e5e7eb",
  dashedBorder: "#60a5fa",
  errorColor: "#dc2626",
  captionBg: "#f1f5f9",
  boxLabelBg: "rgba(0, 0, 0, 0.7)", // Used by canvas now
  boxLabelText: "#ffffff", // Used by canvas now
};

const darkTheme = {
  body: "#0f172a", // Dark blue-gray
  text: "#e2e8f0", // Light gray
  cardBg: "#1e293b", // Slightly lighter dark
  buttonPrimaryBg: "#3b82f6",
  buttonPrimaryHover: "#60a5fa",
  buttonSecondaryBg: "#10b981",
  buttonSecondaryHover: "#34d399",
  accent: "#60a5fa",
  borderColor: "#334155",
  dashedBorder: "#3b82f6",
  errorColor: "#f87171",
  captionBg: "#1e293b",
  boxLabelBg: "rgba(255, 255, 255, 0.7)", // Used by canvas now
  boxLabelText: "#000000", // Used by canvas now
};

// --- Styled Components ---

const Container = styled.div`
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 2rem 1rem; /* Add horizontal padding */
  background: ${({ theme }) => theme.body};
`;

const ThemeToggle = styled.button`
  position: absolute;
  top: 1rem;
  right: 1rem;
  background: none;
  border: none;
  color: ${({ theme }) => theme.text};
  cursor: pointer;
  font-size: 1.6rem;
  padding: 0.5rem;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.2s;

  &:hover {
    background: ${({ theme }) => theme.cardBg};
  }
`;

const Hero = styled.section`
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  max-width: 800px;
  margin-bottom: 2.5rem;

  h1 {
    font-size: clamp(2rem, 6vw, 3rem); /* Responsive font size */
    font-weight: 700;
    margin-bottom: 0.5rem;
    color: ${({ theme }) => theme.text};
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  p {
    font-size: clamp(1rem, 4vw, 1.1rem);
    color: ${({ theme }) =>
      theme.body === lightTheme.body ? "#6b7280" : "#94a3b8"};
    max-width: 600px;
  }
`;

const UploadBox = styled(motion.div)`
  border: 3px dashed ${({ theme }) => theme.dashedBorder};
  padding: clamp(1.5rem, 5vw, 2.5rem);
  border-radius: 1rem;
  text-align: center;
  background-color: ${({ theme }) => theme.cardBg};
  transition: border-color 0.3s, background-color 0.3s;
  margin-bottom: 2rem;
  max-width: 600px;
  width: 95%; /* Responsive width */
  cursor: pointer;
  position: relative; /* Needed for overlay */

  p {
    margin: 0.5rem 0 1rem 0;
    color: ${({ theme }) =>
      theme.body === lightTheme.body ? "#4b5563" : "#cbd5e1"};
  }

  /* Highlight on drag over */
  ${({ isDraggingOver }) =>
    isDraggingOver &&
    css`
      border-color: ${({ theme }) => theme.accent};
      background-color: ${({ theme }) =>
        theme.body === lightTheme.body ? "#eff6ff" : "#1e3a8a"};
    `}
`;

const UploadIconContainer = styled.div`
  margin-bottom: 1rem;
  color: ${({ theme }) => theme.accent};
`;

const UploadLabel = styled.label`
  background-color: ${({ theme }) => theme.buttonPrimaryBg};
  color: white;
  padding: 0.7rem 1.5rem;
  border-radius: 9999px;
  font-weight: 600;
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
  transition: background-color 0.3s, transform 0.1s;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);

  &:hover {
    background-color: ${({ theme }) => theme.buttonPrimaryHover};
  }
  &:active {
    transform: scale(0.98);
  }
`;

const FileName = styled.p`
  margin-top: 1rem !important;
  font-size: 0.9rem;
  color: ${({ theme }) =>
    theme.body === lightTheme.body ? "#374151" : "#d1d5db"};
  word-break: break-all;
`;

const spin = keyframes`
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
`;

const Spinner = styled.div`
  border: 3px solid
    ${({ theme }) => (theme.body === lightTheme.body ? "#e5e7eb" : "#4b5563")};
  border-top: 3px solid ${({ theme }) => theme.accent};
  border-radius: 50%;
  width: 20px;
  height: 20px;
  animation: ${spin} 0.8s linear infinite;
`;

const ButtonGroup = styled.div`
  display: flex;
  flex-wrap: wrap; /* Allow buttons to wrap on smaller screens */
  justify-content: center;
  gap: 1rem;
  margin: 1.5rem 0;
`;

const ActionButton = styled.button`
  background-color: ${({ theme, secondary }) =>
    secondary ? theme.buttonSecondaryBg : theme.buttonPrimaryBg};
  color: white;
  padding: 0.7rem 1.5rem;
  border-radius: 9999px;
  border: none;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 0.6rem;
  transition: background-color 0.3s, transform 0.1s, box-shadow 0.2s;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
  font-size: 0.95rem;

  &:hover {
    background-color: ${({ theme, secondary }) =>
      secondary ? theme.buttonSecondaryHover : theme.buttonPrimaryHover};
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  &:active:not(:disabled) {
    transform: scale(0.98);
  }

  /* Style spinner inside button */
  ${Spinner} {
    width: 18px;
    height: 18px;
  }
`;

const ErrorMessage = styled(motion.p)`
  color: ${({ theme }) => theme.errorColor};
  background-color: ${({ theme }) =>
    theme.body === lightTheme.body ? "#fee2e2" : "#3f1a1a"};
  border: 1px solid ${({ theme }) => theme.errorColor};
  text-align: center;
  font-weight: 500;
  padding: 0.8rem 1.2rem;
  border-radius: 0.5rem;
  margin: 1.5rem auto;
  max-width: 600px;
  width: 95%;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  justify-content: center;
`;

const ResultCard = styled(motion.div)`
  background: ${({ theme }) => theme.cardBg};
  border: 1px solid ${({ theme }) => theme.borderColor};
  border-radius: 1rem;
  padding: 1.5rem;
  margin-top: 1.5rem;
  max-width: 600px;
  width: 95%;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);

  h3 {
    margin-top: 0;
    margin-bottom: 1rem;
    color: ${({ theme }) => theme.accent};
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  p {
    margin-bottom: 1rem;
    line-height: 1.7;
  }
`;

const CaptionResult = styled(ResultCard)`
  background: ${({ theme }) => theme.captionBg};
  border-color: transparent;
`;

const SpeakButton = styled.button`
  background: none;
  color: ${({ theme }) => theme.accent};
  border: none;
  cursor: pointer;
  font-weight: 600;
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  padding: 0.2rem;
  border-radius: 4px;
  transition: color 0.2s, background 0.2s;

  &:hover {
    color: ${({ theme }) => theme.buttonPrimaryHover};
    background: ${({ theme }) => theme.borderColor};
  }
`;

// Container for the original preview image (optional)
const ImagePreviewContainer = styled(motion.div)`
  position: relative;
  margin-bottom: 1rem;
  max-width: 600px;
  width: 95%;
  display: flex;
  justify-content: center;

  img {
    display: block;
    max-width: 100%;
    max-height: 400px;
    border-radius: 0.75rem;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    background-color: ${({ theme }) => theme.borderColor}; /* Placeholder bg */
  }
`;

// --- Helper function to draw boxes on canvas ---
const drawDetectionsOnCanvas = (imageUrl, detections, themeColors) => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous"; // Handle potential CORS issues if image source changes later

    img.onload = () => {
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");

      // Set canvas dimensions to original image dimensions
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;

      // Draw the original image
      ctx.drawImage(img, 0, 0);

      // Prepare drawing styles using theme colors
      ctx.strokeStyle = themeColors.buttonSecondaryBg || "#10b981"; // Use theme color for border
      ctx.lineWidth = Math.max(2, Math.round(img.naturalWidth / 300)); // Dynamic line width
      const fontSize = Math.max(12, Math.round(img.naturalHeight / 50)); // Adjusted font size calculation
      ctx.font = `${fontSize}px Poppins`; // Dynamic font size

      // Draw each detection
      detections.forEach((obj) => {
        const [xmin, ymin, xmax, ymax] = obj.box;
        const width = xmax - xmin;
        const height = ymax - ymin;
        const label = `${obj.label} (${(obj.score * 100).toFixed(1)}%)`;

        // Draw the bounding box
        ctx.strokeRect(xmin, ymin, width, height);

        // --- Draw label with background ---
        const textMetrics = ctx.measureText(label);
        const textWidth = textMetrics.width;
        const textHeight = fontSize; // Use calculated font size for height estimate
        const padding = ctx.lineWidth; // Padding around text

        // Calculate background position (above the box)
        let bgX = xmin - ctx.lineWidth / 2;
        let bgY = ymin - textHeight - padding * 2; // Position above the top border
        let bgWidth = textWidth + padding * 2;
        let bgHeight = textHeight + padding * 2;

        // Adjust if label goes off the top edge
        if (bgY < 0) {
          bgY = ymax + padding; // Place below the bottom border instead
        }
        // Adjust if label goes off the left edge (less common)
        if (bgX < 0) {
          bgX = 0;
        }
        // Adjust if label goes off the right edge
        if (bgX + bgWidth > canvas.width) {
          bgX = canvas.width - bgWidth;
        }

        // Draw background rectangle
        ctx.fillStyle = themeColors.boxLabelBg || "rgba(0, 0, 0, 0.7)";
        ctx.fillRect(bgX, bgY, bgWidth, bgHeight);

        // Draw label text on top of background
        ctx.fillStyle = themeColors.boxLabelText || "#ffffff";
        ctx.fillText(label, bgX + padding, bgY + textHeight + padding / 2); // Adjust Y position for vertical centering
      });

      // Resolve with the new image data URL
      resolve(canvas.toDataURL("image/jpeg")); // Or 'image/png'
    };

    img.onerror = (err) => {
      console.error("Error loading image onto canvas:", err);
      reject(new Error("Could not load image for drawing detections."));
    };

    img.src = imageUrl;
  });
};

// --- App Component ---
function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [caption, setCaption] = useState("");
  const [detectedObjects, setDetectedObjects] = useState([]);
  const [isLoadingCaption, setIsLoadingCaption] = useState(false);
  const [isLoadingDetection, setIsLoadingDetection] = useState(false);
  const [error, setError] = useState(null);
  const [theme, setTheme] = useState("light"); // 'light' or 'dark'
  const [isDraggingOver, setIsDraggingOver] = useState(false);

  // State for the image with boxes drawn on it
  const [drawnImageUrl, setDrawnImageUrl] = useState(null);

  // Ref for original image if needed for display logic
  const imageRef = useRef(null);
  // State for original image dimensions (might be useful, needed by canvas)
  const [originalImageSize, setOriginalImageSize] = useState({
    width: 0,
    height: 0,
  });

  const currentTheme = theme === "light" ? lightTheme : darkTheme;

  // --- Handlers ---

  const clearState = () => {
    setCaption("");
    setDetectedObjects([]);
    setError(null);
    setDrawnImageUrl(null); // Clear the drawn image too
  };

  const handleFileChange = (file) => {
    if (file && file.type.startsWith("image/")) {
      setSelectedFile(file);
      clearState(); // Clear previous results

      const reader = new FileReader();
      reader.onload = (e) => {
        const dataUrl = e.target.result;
        setPreviewUrl(dataUrl); // Set preview immediately

        // Load image to get dimensions *after* setting preview
        const img = new Image();
        img.onload = () => {
          console.log(
            `Original Image Dimensions: ${img.naturalWidth}x${img.naturalHeight}`
          );
          setOriginalImageSize({
            width: img.naturalWidth,
            height: img.naturalHeight,
          });
        };
        img.onerror = () => {
          console.error("Error loading image for dimension check.");
          setError("Could not read image dimensions.");
        };
        img.src = dataUrl; // Load from the Data URL
      };
      reader.onerror = () => {
        console.error("Error reading file.");
        setError("Failed to read the selected file.");
      };
      reader.readAsDataURL(file);
    } else if (file) {
      setError("Invalid file type. Please select an image.");
      setSelectedFile(null);
      setPreviewUrl(null);
    }
  };

  const onFileSelected = (e) => {
    handleFileChange(e.target.files[0]);
    e.target.value = null; // Allow selecting the same file again
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingOver(false);
    handleFileChange(e.dataTransfer.files[0]);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingOver(false);
  };

  const handleGenerateCaption = useCallback(async () => {
    if (!selectedFile) {
      setError("Please select an image file first.");
      return;
    }
    setIsLoadingCaption(true);
    setError(null);
    setCaption("");
    setDrawnImageUrl(null); // Clear detection results when generating caption

    try {
      const generatedCaption = await uploadImageAndGetCaption(selectedFile);
      setCaption(generatedCaption);
    } catch (err) {
      console.error("Caption generation failed:", err);
      setError(err.message || "Failed to generate caption.");
    } finally {
      setIsLoadingCaption(false);
    }
  }, [selectedFile]);

  const handleDetectObjects = useCallback(async () => {
    if (!selectedFile) {
      setError("Please select an image file first.");
      return;
    }
    setIsLoadingDetection(true);
    setError(null);
    setDetectedObjects([]); // Clear previous detections
    setDrawnImageUrl(null); // Clear previous drawn image
    // setCaption(""); // Clear caption results when detecting objects

    try {
      const objects = await uploadImageAndDetectObjects(selectedFile);
      // State will be set, triggering useEffect for drawing
      setDetectedObjects(objects);
      if (objects.length === 0) {
        console.log("No objects detected above the confidence threshold.");
        setError("No objects detected in the image."); // Provide feedback
      }
    } catch (err) {
      console.error("Object detection failed in component:", err);
      setError(err.message || "Failed to detect objects.");
    } finally {
      setIsLoadingDetection(false);
    }
  }, [selectedFile]);

  const speakCaption = () => {
    if (!caption) return;
    try {
      const utterance = new SpeechSynthesisUtterance(caption);
      window.speechSynthesis.speak(utterance);
    } catch (speechError) {
      console.error("Speech synthesis error:", speechError);
      setError(
        "Could not speak the caption. Your browser might not support it."
      );
    }
  };

  // --- useEffect to draw detections when they change ---
  useEffect(() => {
    // Only draw if we have a preview, detections, and original dimensions
    if (
      previewUrl &&
      detectedObjects.length > 0 &&
      originalImageSize.width > 0
    ) {
      console.log("Dependencies met, attempting to draw on canvas...");
      // Pass the current theme colors to the drawing function
      const themeColors = {
        buttonSecondaryBg: currentTheme.buttonSecondaryBg,
        boxLabelBg: currentTheme.boxLabelBg,
        boxLabelText: currentTheme.boxLabelText,
      };
      drawDetectionsOnCanvas(previewUrl, detectedObjects, themeColors)
        .then((dataUrl) => {
          console.log(
            "Canvas drawing successful, updating drawnImageUrl state."
          );
          setDrawnImageUrl(dataUrl);
        })
        .catch((drawError) => {
          console.error("Failed to draw detections:", drawError);
          setError(
            drawError.message || "Failed to render detection boxes on image."
          );
          setDrawnImageUrl(null); // Ensure it's cleared on error
        });
    } else {
      // Clear the drawn image if dependencies are not met
      // Avoid clearing if loading detection to prevent flicker
      if (drawnImageUrl && !isLoadingDetection) {
        console.log(
          "Clearing drawn image URL (dependencies not met or no detections)."
        );
        setDrawnImageUrl(null);
      }
    }
  }, [
    detectedObjects,
    previewUrl,
    originalImageSize,
    currentTheme,
    isLoadingDetection,
    drawnImageUrl,
  ]); // Add currentTheme, isLoadingDetection, drawnImageUrl

  // --- Render ---
  return (
    <ThemeProvider theme={currentTheme}>
      <GlobalStyle theme={currentTheme} />
      <Container>
        {/* --- Theme Toggle --- */}
        <ThemeToggle
          onClick={() => setTheme(theme === "light" ? "dark" : "light")}
          aria-label={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
        >
          {theme === "light" ? <FiMoon /> : <FiSun />}
        </ThemeToggle>

        {/* --- Hero Section --- */}
        <Hero>
          <h1>
            <FiImage aria-hidden="true" /> AI Vision Tools
          </h1>
          <p>
            Upload an image to generate descriptive captions or detect objects
            within it.
          </p>
        </Hero>

        {/* --- Upload Area --- */}
        <UploadBox
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          isDraggingOver={isDraggingOver}
          onClick={() => document.getElementById("imageUpload")?.click()}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <UploadIconContainer>
            <FiUpload size={42} />
          </UploadIconContainer>
          <p>Drag & drop image here, or click to browse</p>
          <UploadLabel htmlFor="imageUpload">Choose Image</UploadLabel>
          <input
            id="imageUpload"
            type="file"
            accept="image/*"
            onChange={onFileSelected}
            style={{ display: "none" }}
            aria-hidden="true"
          />
          {selectedFile && <FileName>{selectedFile.name}</FileName>}
        </UploadBox>

        {/* --- Original Preview (Show only if no result image is ready) --- */}
        <AnimatePresence>
          {previewUrl && (
            <ImagePreviewContainer
              key="original-preview"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ duration: 0.4 }}
              style={{ marginBottom: "1rem" }} // Keep margin for spacing before buttons
              theme={currentTheme} // Pass theme for background color
            >
              <img
                ref={imageRef}
                src={previewUrl}
                alt="Selected preview"
                style={{
                  /* Basic image styles */ display: "block",
                  maxWidth: "100%",
                  maxHeight: "400px",
                  borderRadius: "0.75rem",
                  boxShadow: "0 5px 15px rgba(0,0,0,0.1)",
                  backgroundColor: currentTheme.borderColor,
                }}
              />
            </ImagePreviewContainer>
          )}
        </AnimatePresence>

        {/* --- Action Buttons --- */}
        {selectedFile && (
          <ButtonGroup>
            <ActionButton
              onClick={handleGenerateCaption}
              disabled={isLoadingCaption || isLoadingDetection}
              secondary // Caption button uses secondary color
            >
              {isLoadingCaption ? <Spinner /> : <FiMessageSquare />}
              Generate Caption
            </ActionButton>
            <ActionButton
              onClick={handleDetectObjects}
              disabled={isLoadingCaption || isLoadingDetection}
            >
              {isLoadingDetection ? <Spinner /> : <FiEye />}
              Detect Objects
            </ActionButton>
          </ButtonGroup>
        )}

        {/* --- Error Message --- */}
        <AnimatePresence>
          {error && (
            <ErrorMessage
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              role="alert"
              theme={currentTheme} // Pass theme for colors
            >
              <FiXCircle style={{ marginRight: "0.5rem", flexShrink: 0 }} />{" "}
              {error}
            </ErrorMessage>
          )}
        </AnimatePresence>

        {/* --- Display Drawn Image Result --- */}
        <AnimatePresence>
          {drawnImageUrl && (
            <ResultCard // Use ResultCard for consistent styling
              key="drawn-image"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              theme={currentTheme} // Pass theme
            >
              <h3>
                <FiCheckCircle /> Detection Results
              </h3>
              <img
                src={drawnImageUrl}
                alt="Image with detected objects"
                style={{
                  display: "block",
                  maxWidth: "100%",
                  maxHeight: "500px", // Allow slightly larger display for result
                  borderRadius: "0.5rem", // Match card border radius
                  margin: "0 auto", // Center image within card
                  // Removed redundant border/shadow as it's on the card
                }}
              />
            </ResultCard>
          )}
        </AnimatePresence>

        {/* --- Display Caption Result --- */}
        <AnimatePresence>
          {caption && (
            <CaptionResult
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ type: "spring", stiffness: 100 }}
              theme={currentTheme} // Pass theme
            >
              <h3>
                <FiMessageSquare /> Generated Caption
              </h3>
              <p>{caption}</p>
              <SpeakButton
                onClick={speakCaption}
                aria-label="Speak caption aloud"
                theme={currentTheme}
              >
                <FiVolume2 /> Speak
              </SpeakButton>
            </CaptionResult>
          )}
        </AnimatePresence>
      </Container>
    </ThemeProvider>
  );
}

export default App;
