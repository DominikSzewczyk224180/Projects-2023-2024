# Use an official Nginx image to serve the frontend
FROM nginx:alpine

# Set the working directory inside the container
WORKDIR /usr/share/nginx/html

# Remove the default Nginx index page
RUN rm -rf ./*

# Copy HTML files from the src directory to the working directory
COPY src/ .


# Expose port 80
EXPOSE 80

# Start Nginx server
CMD ["nginx", "-g", "daemon off;"]

# Build and run commands
# docker build -t athena-frontend .
# docker run -p 80:80 athena-frontend
