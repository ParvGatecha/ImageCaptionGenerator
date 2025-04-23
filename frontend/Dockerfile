# ---- Build Stage ----
# Use an official Node runtime as the build environment
FROM node:23-alpine-slim as builder

# Set the working directory
WORKDIR /app

# Copy package.json and package-lock.json (or yarn.lock)
COPY package*.json ./

# Install dependencies
# Use ci for reproducible installs, fallback to install if no lockfile
RUN npm ci || npm install

# Copy the rest of the application code
COPY . .

# Build the React application for production
RUN npm run build

# ---- Run Stage ----
# Use a lightweight Nginx image to serve the static files
FROM nginx:1.27-alpine-slim

# Copy the Nginx configuration file
# This file tells Nginx how to serve the React app and potentially proxy API requests
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy the built static files from the builder stage to Nginx's web root
COPY --from=builder /app/build /usr/share/nginx/html

# Expose port 80 (Nginx default)
EXPOSE 80

# Command to run Nginx in the foreground
CMD ["nginx", "-g", "daemon off;"]