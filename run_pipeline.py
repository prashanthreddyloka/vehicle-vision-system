"""
Simple script to run the vehicle analysis pipeline on your video.
Supports both local files and video URLs (YouTube, direct video links).
"""

from main import VehicleAnalysisPipeline
import os
import urllib.request
import urllib.parse

def download_video(url, output_path="data/downloaded_video.mp4"):
    """Download video from URL."""
    print(f"\n📥 Downloading video from URL...")
    print(f"   URL: {url}")
    
    try:
        # Check if it's a YouTube URL
        if 'youtube.com' in url or 'youtu.be' in url:
            print("\n⚠️  YouTube URL detected!")
            print("   You need 'yt-dlp' or 'youtube-dl' installed to download YouTube videos.")
            print("   Install with: pip install yt-dlp")
            
            try:
                import yt_dlp
                
                ydl_opts = {
                    'format': 'best[ext=mp4]',
                    'outtmpl': output_path,
                    'quiet': False,
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                
                print(f"✅ Downloaded to: {output_path}")
                return output_path
                
            except ImportError:
                print("\n❌ yt-dlp not installed!")
                print("   Run: .\\venv\\Scripts\\pip.exe install yt-dlp")
                return None
        
        # Direct video URL (mp4, avi, etc.)
        else:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            
            # Download with progress
            def progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, block_num * block_size * 100 / total_size)
                    print(f"\r   Progress: {percent:.1f}%", end='', flush=True)
            
            urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
            print(f"\n✅ Downloaded to: {output_path}")
            return output_path
            
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return None

def is_url(path):
    """Check if the input is a URL."""
    return path.startswith('http://') or path.startswith('https://')

def main():
    print("=" * 70)
    print("  Vehicle License Plate Scanner & Classification System")
    print("=" * 70)
    print()
    
    # Get input video path or URL from user
    print("Please provide the path or URL to your input video:")
    print()
    print("Examples:")
    print("  Local file: C:\\Users\\prash\\Videos\\traffic.mp4")
    print("  Direct URL: https://example.com/traffic.mp4")
    print("  YouTube:    https://www.youtube.com/watch?v=...")
    print()
    video_input = input("Input (path or URL): ").strip().strip('"')
    
    # Handle URL input
    if is_url(video_input):
        downloaded_path = download_video(video_input, "data/downloaded_video.mp4")
        if not downloaded_path:
            print("\n❌ Could not download video. Exiting.")
            return
        video_input = downloaded_path
    
    # Check if file exists
    if not os.path.exists(video_input):
        print(f"\n❌ Error: File '{video_input}' not found!")
        print("\nMake sure to provide the full path to your video file.")
        return
    
    print(f"\n✅ Video ready: {video_input}")
    
    # Ask for output location
    print("\nWhere should I save the processed video?")
    default_output = "data/output_video.mp4"
    video_output = input(f"Output video path (press Enter for '{default_output}'): ").strip().strip('"')
    if not video_output:
        video_output = default_output
    
    # Ask for CSV output location
    print("\nWhere should I save the results CSV?")
    default_csv = "data/results.csv"
    csv_output = input(f"CSV output path (press Enter for '{default_csv}'): ").strip().strip('"')
    if not csv_output:
        csv_output = default_csv
    
    # Ensure CSV has .csv extension
    if not csv_output.lower().endswith('.csv'):
        csv_output = csv_output + '.csv'
        print(f"   Added .csv extension: {csv_output}")
    
    # Ask if they want to limit frames (for testing)
    print("\nDo you want to process the entire video or just test with a few frames?")
    max_frames_input = input("Max frames to process (press Enter for entire video): ").strip()
    max_frames = None
    if max_frames_input:
        try:
            max_frames = int(max_frames_input)
        except ValueError:
            print("Invalid number, processing entire video...")
    
    # Create output directories if needed
    video_dir = os.path.dirname(video_output)
    if video_dir and video_dir != '.':
        os.makedirs(video_dir, exist_ok=True)
    
    csv_dir = os.path.dirname(csv_output)
    if csv_dir and csv_dir != '.':
        os.makedirs(csv_dir, exist_ok=True)
    
    # Run the pipeline
    print("\n" + "=" * 70)
    print("Starting processing...")
    print("=" * 70)
    print(f"Input:  {video_input}")
    print(f"Output: {video_output}")
    print(f"CSV:    {csv_output}")
    if max_frames:
        print(f"Frames: {max_frames} (testing mode)")
    else:
        print(f"Frames: All (full video)")
    print("=" * 70)
    print()
    
    # Initialize and run pipeline
    pipeline = VehicleAnalysisPipeline()
    pipeline.process_video(video_input, video_output, max_frames=max_frames)
    pipeline.save_results(csv_output)
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ Processing Complete!")
    print("=" * 70)
    print(f"\n📹 Annotated video saved to: {video_output}")
    print(f"📊 Results CSV saved to: {csv_output}")
    
    if pipeline.results:
        unique_vehicles = len(set([r['vehicle_id'] for r in pipeline.results]))
        plates_detected = sum(1 for r in pipeline.results if r['license_plate'] != "N/A")
        print(f"\n📈 Summary:")
        print(f"   - Total detections: {len(pipeline.results)}")
        print(f"   - Unique vehicles: {unique_vehicles}")
        print(f"   - License plates read: {plates_detected}")
    
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing cancelled by user.")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
