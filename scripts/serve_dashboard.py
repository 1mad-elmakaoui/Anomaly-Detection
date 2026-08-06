"""
Simple HTTP server for the dashboard.
Run this to avoid CORS issues when loading the dashboard locally.
"""

import http.server
import socketserver
import os
from pathlib import Path

# Get the dashboard directory
DASHBOARD_DIR = Path(__file__).parent.parent / 'dashboard'

class NoCacheHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler with no caching."""
    
    def end_headers(self):
        # Disable caching
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        self.send_header('Expires', '0')
        super().end_headers()

def main():
    PORT = 8000
    
    # Change to dashboard directory
    os.chdir(DASHBOARD_DIR)
    
    # Create server
    with socketserver.TCPServer(("", PORT), NoCacheHTTPRequestHandler) as httpd:
        print(f"Dashboard running at http://localhost:{PORT}")
        print(f"Serving files from {DASHBOARD_DIR}")
        print("\nPress Ctrl+C to stop the server\n")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nServer stopped.")

if __name__ == '__main__':
    main()
