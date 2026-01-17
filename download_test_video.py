"""
Download sample traffic video for pipeline verification.
"""
import urllib.request
import os

def download_sample_video():
    """Download a sample traffic video from a public source."""
    # Sample video URLs (public domain or CC0)
    sample_urls = [
        # Pexels free video (cars on highway)
        "https://player.vimeo.com/external/371433846.sd.mp4?s=236a1e3f3c7f1f7e5c3b3e8d7e6f5c4b3a2a1a0a&profile_id=165&oauth2_token_id=57447761",
    ]
    
    output_path = "data/test_traffic.mp4"
    
    print("Downloading sample traffic video for testing...")
    print("This may take a moment...")
    
    try:
        urllib.request.urlretrieve(sample_urls[0], output_path)
        print(f"✓ Sample video downloaded: {output_path}")
        return output_path
    except Exception as e:
        print(f"✗ Failed to download: {e}")
        return None

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    download_sample_video()
