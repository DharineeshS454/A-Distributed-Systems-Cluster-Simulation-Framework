import requests
import time
import sys
import uuid
import logging
import signal
import os
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Node:
    def __init__(self, node_id, cpu_cores):
        self.node_id = node_id
        self.cpu_cores = int(cpu_cores)
        self.available_cores = self.cpu_cores
        self.pods = []  # List to store pod IDs
        
        # Get API server URL from environment variable or use default
        self.api_server_url = os.environ.get("API_SERVER_URL", "http://localhost:5001")
        logger.info(f"Using API server URL: {self.api_server_url}")
        
        self.running = True

    def register(self):
        """Register the node with the API server"""
        try:
            response = requests.post(
                f"{self.api_server_url}/nodes",
                json={
                    'node_id': self.node_id,
                    'cpu_cores': self.cpu_cores
                }
            )
            if response.status_code in [200, 201]:
                # Assume the server returns the node_id (or uuid) on successful registration
                self.node_id = response.json().get('node_id', self.node_id)
                logger.info(f"Node {self.node_id}: Registered successfully with the API server.")
            else:
                logger.error(f"Node {self.node_id}: Failed to register - {response.status_code}: {response.text}")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Node {self.node_id}: Error during registration - {str(e)}")
            sys.exit(1)

    def send_heartbeat(self):
        """Send heartbeat signal to the API server"""
        try:
            response = requests.post(
                f"{self.api_server_url}/nodes/{self.node_id}/heartbeat",
                json={
                    'status': 'healthy',
                    'available_cores': self.available_cores,
                    'pods': self.pods
                }
            )
            
            if response.status_code == 200:
                logger.debug(f"Node {self.node_id}: Heartbeat sent successfully")
            else:
                logger.warning(f"Node {self.node_id}: Failed to send heartbeat - {response.status_code}: {response.text}")
                
        except Exception as e:
            logger.error(f"Node {self.node_id}: Error sending heartbeat - {str(e)}")

    def start_heartbeat_loop(self):
        """Start sending periodic heartbeats to the API server"""
        logger.info(f"Node {self.node_id}: Starting heartbeat loop")
        
        while self.running:
            self.send_heartbeat()
            time.sleep(5)  # Send heartbeat every 5 seconds

    def handle_shutdown(self, signum, frame):
        """Handle graceful shutdown"""
        logger.info(f"Node {self.node_id}: Shutting down...")
        self.running = False

def main():
    # Set up argparse to handle command-line arguments
    parser = argparse.ArgumentParser(description="Simulate a node in a Kubernetes-like cluster")
    parser.add_argument("--node-id", type=str, required=True, help="ID of the node")
    parser.add_argument("--cpu-cores", type=int, required=True, help="Number of CPU cores available on the node")
    parser.add_argument("--api-server-url", type=str, default="http://127.0.0.1:5001", help="API server URL")

    # Parse the arguments
    args = parser.parse_args()

    # Create and start the node
    node = Node(args.node_id, args.cpu_cores)
    
    logger.info(f"Node {args.node_id} started with {args.cpu_cores} CPU cores")
    
    # Register the node with the API server
    node.register()

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, node.handle_shutdown)
    signal.signal(signal.SIGINT, node.handle_shutdown)
    
    # Start sending heartbeats
    node.start_heartbeat_loop()

if __name__ == "__main__":
    main()
