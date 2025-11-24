from pytubefix import YouTube
import cv2
import numpy as np
from pathlib import Path
from global_stuff.constants import BASE_DIR
from global_stuff.crop_arena import crop_single_image
from PIL import Image
def extract_frames_from_video(paths, output_dir, total_saved, every_n_frames=30*15, crop_arena=True):
    output_dir.mkdir(exist_ok=True)
    for video_path in paths:
        print("Extracting frames...")
        cap = cv2.VideoCapture(BASE_DIR / video_path)
        count=0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                count += 1
                continue  # Skip to next frame
            if not ret:
                break
            if count % every_n_frames == 0:
                frame_path = output_dir / f"frame_{total_saved:04}.jpg"
                if crop_arena:
                    cropped = crop_single_image(frame)
                    frame = cv2.resize(cropped, (420,896), interpolation=cv2.INTER_LANCZOS4)
                    cv2.imwrite(str(frame_path), cropped)
                else:
                    cv2.imwrite(str(frame_path), frame)
                total_saved += 1
            count+=1

        cap.release()
        Path(video_path).unlink()
    return total_saved
def extract_frames_for_analysis(path, output_dir):
    print(f"Extracting frames from {path} to {output_dir}")
    Path(output_dir).mkdir(exist_ok=True,parents=True)
    cap = cv2.VideoCapture(path)
    count=0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = output_dir / f"frame_{count:04}.png"
        try:
            print (f"frame_path {frame_path}")
            cropped = crop_single_image(frame)
            cv2.imwrite(str(frame_path), cropped)
        except e:
            print("exception {e}")
        count+=1

    cap.release()

def extract_frames_from_youtube(urls):
    output_dir = BASE_DIR / "frames" # CHANGE DEPDNING ON USE CASE
    output_dir.mkdir(parents=True, exist_ok=True)
    total_saved = 0

    for idx, url in enumerate(urls):
        print(f"\n📥 Processing video {idx+1}/{len(urls)}: {url}")
        try:
            yt = YouTube(url)
            stream = yt.streams.filter(only_video=True, file_extension="mp4").order_by("resolution").desc().first()
            if not stream:
                print("⚠️ No video-only stream found. Skipping.")
                continue

            print(f"Selected stream: {stream.resolution} ({stream.mime_type})")

            video_path = f"vid_{idx}.mp4"
            stream.download(filename=video_path)

            total_saved = extract_frames_from_video([video_path], output_dir, total_saved, every_n_frames=15)

            print(f"✅ Finished video {idx+1}, total frames saved so far: {total_saved}")

        except Exception as e:
            print(f"Failed to process video {url}: {e}")

    print(f"\nAll done! Total saved frames: {total_saved}")

if __name__ == "__main__":
    extract_frames_from_youtube(["https://youtu.be/StLHbCmIQQE"])
    #extract_frames_from_video(["vid.mov"], BASE_DIR/"frames", 0, 5)
    #extract_frames_from_video(["replay.MP4"], BASE_DIR/"frames", 0, 10)
    #extract_frames_from_video(["vid_0.mp4"], BASE_DIR/"frames", 0, 30)
    #yt = YouTube("https://youtu.be/VMj-3S1tku0")
    #stream = yt.streams.get_highest_resolution()
    #stream.download()

