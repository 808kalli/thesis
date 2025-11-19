"""
debug_episode_extraction.py

Debug script to test frame extraction from specific episodes
"""

import subprocess
from pathlib import Path
from PIL import Image
from io import BytesIO
import argparse


def test_frame_extraction(dataset_dir, episode_idx, frame_idx=0):
    """Test if we can extract a frame from a specific episode."""
    dataset_dir = Path(dataset_dir)

    # Find video file
    video_path = None
    videos_dir = dataset_dir / 'videos'

    print(f"Looking for episode {episode_idx} video...")
    for chunk_dir in sorted(videos_dir.glob('chunk-*')):
        image_dir = chunk_dir / 'observation.images.image'
        if image_dir.exists():
            potential_video = image_dir / f'episode_{episode_idx:06d}.mp4'
            if potential_video.exists():
                video_path = potential_video
                print(f"✅ Found: {video_path}")
                break

    if not video_path:
        print(f"❌ Video not found for episode {episode_idx}")
        return False

    # Try to extract a frame
    print(f"\nExtracting frame {frame_idx}...")
    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-vf', f'select=eq(n\\,{frame_idx})',
        '-vframes', '1',
        '-f', 'image2pipe',
        '-pix_fmt', 'rgb24',
        '-'
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=5)

        if result.returncode != 0:
            print(f"❌ ffmpeg failed with code {result.returncode}")
            print(f"   stderr: {result.stderr.decode()[:200]}")
            return False

        # Try to open as image
        frame = Image.open(BytesIO(result.stdout)).convert('RGB')
        print(f"✅ Frame extracted successfully: {frame.size}")
        return True

    except subprocess.TimeoutExpired:
        print(f"❌ ffmpeg timeout")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug episode extraction")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--episode_start", type=int, default=295, help="Start episode")
    parser.add_argument("--episode_end", type=int, default=310, help="End episode")

    args = parser.parse_args()

    print(f"Testing episodes {args.episode_start}-{args.episode_end}")
    print("="*70)

    success_count = 0
    fail_count = 0

    for ep in range(args.episode_start, args.episode_end + 1):
        if test_frame_extraction(args.dataset_dir, ep):
            success_count += 1
        else:
            fail_count += 1
        print()

    print("="*70)
    print(f"Results: {success_count} success, {fail_count} failed")
