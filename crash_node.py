import requests
import time
import sys
import logging
import signal
import os
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CrashNode:
    def __init__(self, node_id, cpu_cores):
        self.node_id = node_id
        self.cpu_cores = int(cpu_cores)
        self.available_cores = self.cpu_cores
        self.pods = []
        self.api_server_url = "http://localhost:5001"
        self.running = True

    def register(self):
        try:
            response = requests.post(
                f"{self.api_server_url}/nodes",
                json={
                    'node_id': self.node_id,
                    'cpu_cores': self.cpu_cores
                }
            )
            if response.status_code in [200, 201]:
                logger.info(f"Node {self.node_id}: Registered successfully")
            else:
                logger.error(f"Node {self.node_id}: Failed to register")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Node {self.node_id}: Error during registration - {str(e)}")
            sys.exit(1)

    def send_heartbeat(self):
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
                logger.warning(f"Node {self.node_id}: Failed to send heartbeat")
        except Exception as e:
            logger.error(f"Node {self.node_id}: Error sending heartbeat - {str(e)}")

    def start_heartbeat_loop(self):
        logger.info(f"Node {self.node_id}: Starting heartbeat loop")
        while self.running:
            self.send_heartbeat()
            time.sleep(5)

    def crash_after_delay(self, delay):
        time.sleep(delay)
        logger.info(f"Node {self.node_id}: Simulating crash after {delay} seconds")
        sys.exit(1)  # Force crash the process

def main():
    node = CrashNode("crash-test-node", 2)
    node.register()
    
    # Start heartbeat loop in a separate thread
    heartbeat_thread = threading.Thread(target=node.start_heartbeat_loop)
    heartbeat_thread.daemon = True
    heartbeat_thread.start()
    
    # Start crash timer in a separate thread
    crash_thread = threading.Thread(target=node.crash_after_delay, args=(10,))
    crash_thread.daemon = True
    crash_thread.start()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        node.running = False

if __name__ == "__main__":
    main() 