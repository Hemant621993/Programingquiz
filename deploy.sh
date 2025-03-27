#!/bin/bash

# Exit on error
set -e

# Color codes for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Programming Quiz Application Deployment Script${NC}"
echo -e "${YELLOW}======================================================${NC}"

# Check for Docker and Docker Compose
echo -e "${BLUE}Checking for Docker and Docker Compose...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker not found. Please install Docker first.${NC}"
    echo "Visit https://docs.docker.com/get-docker/ for installation instructions."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose not found. Please install Docker Compose first.${NC}"
    echo "Visit https://docs.docker.com/compose/install/ for installation instructions."
    exit 1
fi

echo -e "${GREEN}Docker and Docker Compose are installed.${NC}"

# Check for .env file and create if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${YELLOW}No .env file found. Creating from template...${NC}"
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${GREEN}.env file created from .env.example${NC}"
        echo -e "${YELLOW}Please edit the .env file to add your API keys before continuing.${NC}"
        exit 0
    else
        echo -e "${RED}.env.example file not found. Please create a .env file manually.${NC}"
        exit 1
    fi
fi

# Prompt for confirmation to continue
echo -e "${YELLOW}This script will deploy the Programming Quiz Application using Docker.${NC}"
echo -e "${YELLOW}Make sure you have edited the .env file with your API keys.${NC}"
read -p "Continue with deployment? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Deployment cancelled.${NC}"
    exit 1
fi

# Start the deployment
echo -e "${BLUE}Building and starting Docker containers...${NC}"
docker-compose up --build -d

# Check if containers are running
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Deployment successful!${NC}"
    echo -e "${GREEN}The application is now running at: http://localhost:5000${NC}"
    echo -e "${BLUE}You can view logs with: docker-compose logs -f${NC}"
    echo -e "${BLUE}To stop the application: docker-compose down${NC}"
else
    echo -e "${RED}Deployment failed. Check the logs for more information.${NC}"
    echo -e "${BLUE}View logs with: docker-compose logs${NC}"
fi