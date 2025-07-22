#!/usr/bin/env python3

"""
Simple Server Validation

This module validates that the model server is running and returns a valid response.

Author: Mi Yan
License: CC-BY-NC 4.0
Created: 2025-07-10
"""

import zmq
import numpy as np
from termcolor import colored

def validate_server(host: str = "127.0.0.1", port: int = 8000, timeout: int = 5) -> bool:
    """
    Validate that the server is running and returns a valid dict.
    
    Args:
        host: Server hostname
        port: Server port
        timeout: Timeout in seconds
        
    Returns:
        True if server returns valid dict, False otherwise
    """
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, timeout * 1000)
    
    try:
        socket.connect(f"tcp://{host}:{port}")
        
        # Create test data matching agent.py format
        mock_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        mock_proprio = [np.random.randn(7) for _ in range(4)]
        
        test_data = {
            'front_view_image': [mock_image],
            'side_view_image': [mock_image],
            'proprio_array': mock_proprio,
            'text': 'Validation test instruction',
        }
        
        socket.send_pyobj(test_data)
        response = socket.recv_pyobj()
        
        # Check if response is a valid dict
        if not isinstance(response, dict):
            print(f"✗ Server returned {type(response)}, expected dict")
            return False
            
        print(colored(f"✓ Server at {host}:{port} returned valid dict", 'green'))
        return True
        
    except zmq.Again:
        print(colored(f"✗ Server at {host}:{port} timeout after {timeout}s", 'red'))
        return False
    except Exception as e:
        print(colored(f"✗ Error connecting to server at {host}:{port}: {e}", 'red'))
        return False
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate model server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=6666, help="Server port")
    parser.add_argument("--timeout", type=int, default=5, help="Timeout in seconds")
    
    args = parser.parse_args()
    validate_server(args.host, args.port, args.timeout)
