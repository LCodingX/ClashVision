import argparse
from pytubefix import YouTube
from pathlib import Path

def download_video(url: str, output_path: str, filename: str):
    try:
        # Use the WEB client and enable PoToken support (requires Node.js ≥ v16)
        yt = YouTube(url, client='WEB', use_po_token=True)

        # Get the best video-only stream
        stream = yt.streams.filter(only_video=True, file_extension="mp4") \
                           .order_by("resolution") \
                           .desc() \
                           .first()

        if stream is None:
            print("❌ No suitable MP4 stream found.")
            return

        print(f"⬇️ Downloading: {yt.title}")
        print(f"📺 Resolution: {stream.resolution}, ⏱ FPS: {stream.fps}, 🎞 Type: {stream.mime_type}")
        Path(output_path).mkdir(parents=True, exist_ok=True)
        stream.download(output_path=output_path, filename=filename)
        print(f"✅ Saved to: {Path(output_path) / filename}")

    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("💡 Try installing Node.js, or using yt-dlp as a fallback.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download high-quality MP4 video using PoToken-enabled PyTubeFix")
    parser.add_argument("--url", type=str, required=True, help="YouTube video URL")
    parser.add_argument("--output", type=str, default="downloads", help="Output directory")
    parser.add_argument("--filename", type=str, default="video.mp4", help="Output filename (with .mp4 extension)")

    args = parser.parse_args()
    download_video(args.url, args.output, args.filename)
