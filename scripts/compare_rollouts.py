#!/usr/bin/env python3
"""
Side-by-side rollout comparison viewer.

Displays videos from two training pipelines (distilled vs vanilla) side-by-side.
Navigate between episodes using arrow keys:
  - Right arrow: Next video pair
  - Left arrow: Previous video pair
  - Space: Pause/Resume playback
  - q or Esc: Quit

Usage:
    python compare_rollouts.py --distilled_dir <path1> --vanilla_dir <path2>
    python compare_rollouts.py <path1> <path2>  # Positional arguments also work
"""

import cv2
import os
import glob
import re
import argparse
from pathlib import Path
import numpy as np

class RolloutComparer:
    def __init__(self, distilled_dir, vanilla_dir):
        """Initialize the rollout comparer with paths to the two rollout directories."""
        self.distilled_dir = Path(distilled_dir)
        self.vanilla_dir = Path(vanilla_dir)

        # Get all videos from both directories
        distilled_files = glob.glob(str(self.distilled_dir / "*.mp4"))
        vanilla_files = glob.glob(str(self.vanilla_dir / "*.mp4"))

        # Build dictionaries mapping episode number to video path
        self.distilled_dict = {}
        for video in distilled_files:
            ep_num = self.extract_episode_number(video)
            if ep_num is not None:
                self.distilled_dict[ep_num] = video

        self.vanilla_dict = {}
        for video in vanilla_files:
            ep_num = self.extract_episode_number(video)
            if ep_num is not None:
                self.vanilla_dict[ep_num] = video

        # Find common episode numbers
        common_episodes = sorted(set(self.distilled_dict.keys()) & set(self.vanilla_dict.keys()))

        if not common_episodes:
            raise ValueError(f"No matching episodes found between directories.\n"
                           f"Distilled episodes: {sorted(self.distilled_dict.keys())[:10]}...\n"
                           f"Vanilla episodes: {sorted(self.vanilla_dict.keys())[:10]}...")

        # Create paired video list based on common episodes
        self.video_pairs = [(self.distilled_dict[ep], self.vanilla_dict[ep]) for ep in common_episodes]

        print(f"Found {len(self.distilled_dict)} distilled videos")
        print(f"Found {len(self.vanilla_dict)} vanilla videos")
        print(f"Matched {len(self.video_pairs)} common episodes")

        # Build task description to ID mapping
        self.task_to_id = self._build_task_mapping()

        self.current_index = 0
        self.window_name = "Rollout Comparison (Distilled | Vanilla) - LEFT/RIGHT to navigate, SPACE to pause, Q to quit"

        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def _get_episode_from_filename(self, filepath):
        """Extract episode number and task for sorting."""
        basename = os.path.basename(filepath)
        try:
            # Extract episode number
            episode_str = basename.split("episode=")[1].split("--")[0]
            episode = int(episode_str)

            # Extract task description for secondary sort
            task_match = re.search(r'task=(.+?)\.mp4', basename)
            task_desc = task_match.group(1).lower().replace('_', ' ') if task_match else ""

            # Return tuple: (task_desc, episode) for sorting
            return (task_desc, episode)
        except:
            return (chr(255), float('inf'))  # Put files without proper format at the end

    def _build_task_mapping(self):
        """Build mapping of task descriptions to task IDs (1-10)."""
        task_descriptions = set()
        for video in self.distilled_dict.values():
            basename = os.path.basename(video)
            task_match = re.search(r'task=(.+?)\.mp4', basename)
            if task_match:
                task_desc = task_match.group(1).lower().replace('_', ' ')
                task_descriptions.add(task_desc)

        # Create mapping: sorted task descriptions to 1-indexed task IDs
        task_to_id = {}
        for idx, task_desc in enumerate(sorted(task_descriptions), 1):
            task_to_id[task_desc] = str(idx)

        return task_to_id

    def extract_episode_number(self, filename):
        """Extract episode number from filename."""
        basename = os.path.basename(filename)
        try:
            # Format: 2025_11_22-16_03_44--episode=123--...
            episode_str = basename.split("episode=")[1].split("--")[0]
            return int(episode_str)
        except:
            return None

    def find_matching_videos(self, index):
        """Get the video pair at the given index."""
        if index < len(self.video_pairs):
            return self.video_pairs[index]
        return None, None

    def get_video_info(self, video_path):
        """Get video information."""
        basename = os.path.basename(video_path)

        # Extract success status (look for True or False followed by --)
        success_match = re.search(r'success=(True|False)--', basename)
        success = success_match.group(1) if success_match else "?"

        # Extract global episode number
        episode_match = re.search(r'episode=(\d+)--', basename)
        if episode_match:
            global_episode = int(episode_match.group(1))
            # Calculate task ID (0-indexed task, then convert to 1-indexed)
            # Each task has 50 episodes: episodes 0-49 = task 0, 50-99 = task 1, etc.
            task_id = (global_episode // 50) + 1  # 1-indexed task ID
            # Calculate local episode within task (1-indexed)
            local_episode = (global_episode % 50) + 1  # 1-indexed episode within task
        else:
            task_id = "?"
            local_episode = "?"

        return str(local_episode), success, str(task_id)

    def get_task_id(self, task_description):
        """Map task description to task ID using the pre-built mapping."""
        task_desc_lower = task_description.lower().replace('_', ' ')
        return self.task_to_id.get(task_desc_lower, "?")

    def display_videos(self):
        """Display videos side by side."""
        distilled_video, vanilla_video = self.find_matching_videos(self.current_index)

        distilled_ep, distilled_success, distilled_task_id = self.get_video_info(distilled_video)
        if vanilla_video:
            vanilla_ep, vanilla_success, vanilla_task_id = self.get_video_info(vanilla_video)
        else:
            vanilla_ep, vanilla_success, vanilla_task_id = "N/A", "N/A", "?"

        # Open video captures
        cap_distilled = cv2.VideoCapture(distilled_video)
        cap_vanilla = cv2.VideoCapture(vanilla_video) if vanilla_video else None

        # Get video properties
        width_distilled = int(cap_distilled.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_distilled = int(cap_distilled.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_distilled = cap_distilled.get(cv2.CAP_PROP_FPS)

        if cap_vanilla:
            width_vanilla = int(cap_vanilla.get(cv2.CAP_PROP_FRAME_WIDTH))
            height_vanilla = int(cap_vanilla.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_vanilla = cap_vanilla.get(cv2.CAP_PROP_FPS)

        # Calculate frame delay (in ms)
        frame_delay = int(1000 / fps_distilled) if fps_distilled > 0 else 30

        # Play videos
        paused = False
        while True:
            if not paused:
                ret_distilled, frame_distilled = cap_distilled.read()
                ret_vanilla = cap_vanilla.read() if cap_vanilla else (None, None)
                ret_vanilla, frame_vanilla = ret_vanilla

                # Stop if either video ends
                if not ret_distilled or (cap_vanilla and not ret_vanilla):
                    break

                # Resize frames to same height for side-by-side display (doubled size)
                target_height = 1440
                scale_distilled = target_height / height_distilled
                new_width_distilled = int(width_distilled * scale_distilled)
                frame_distilled_resized = cv2.resize(frame_distilled, (new_width_distilled, target_height))

                if frame_vanilla is not None:
                    scale_vanilla = target_height / height_vanilla
                    new_width_vanilla = int(width_vanilla * scale_vanilla)
                    frame_vanilla_resized = cv2.resize(frame_vanilla, (new_width_vanilla, target_height))

                    # Combine frames side by side
                    combined = np.hstack([frame_distilled_resized, frame_vanilla_resized])

                    # Add top margin for labels
                    title_height = 100
                    combined_with_title = np.ones((combined.shape[0] + title_height, combined.shape[1], 3), dtype=np.uint8) * 255
                    combined_with_title[title_height:, :] = combined

                    # DISTILLED label (left side) - white color
                    cv2.putText(combined_with_title, "DISTILLED", (60, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 5)

                    # VANILLA label (right side) - white color
                    cv2.putText(combined_with_title, "VANILLA", (new_width_distilled + 60, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 5)

                    # Store for redisplay when paused
                    self.last_frame = combined_with_title
                else:
                    # Only distilled video, add label bar to it
                    title_height = 100
                    combined_with_title = np.ones((frame_distilled_resized.shape[0] + title_height, frame_distilled_resized.shape[1], 3), dtype=np.uint8) * 255
                    combined_with_title[title_height:, :] = frame_distilled_resized

                    # DISTILLED label
                    cv2.putText(combined_with_title, "DISTILLED", (60, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 5)

                    # Store for redisplay when paused
                    self.last_frame = combined_with_title

            cv2.imshow(self.window_name, self.last_frame)

            # Handle keyboard input
            key = cv2.waitKey(frame_delay) & 0xFF

            if key == ord('q') or key == 27:  # 27 is Esc
                cap_distilled.release()
                if cap_vanilla:
                    cap_vanilla.release()
                cv2.destroyWindow(self.window_name)
                return False
            elif key == ord(' '):  # Space bar - toggle pause
                paused = not paused
            elif key == 83 or key == 2555904:  # Right arrow (83 or specific code depending on system)
                # Move to next video
                cap_distilled.release()
                if cap_vanilla:
                    cap_vanilla.release()

                if self.current_index < len(self.video_pairs) - 1:
                    self.current_index += 1
                    return True
                else:
                    print("Already at the last video pair")
                    return False
            elif key == 81 or key == 2424832:  # Left arrow (81 or specific code)
                # Move to previous video
                cap_distilled.release()
                if cap_vanilla:
                    cap_vanilla.release()

                if self.current_index > 0:
                    self.current_index -= 1
                    return True
                else:
                    print("Already at the first video pair")
                    return False

        cap_distilled.release()
        if cap_vanilla:
            cap_vanilla.release()

        return True

    def run(self):
        """Run the comparison viewer."""
        while True:
            should_continue = self.display_videos()
            if not should_continue:
                break

def print_eval_results():
    """Print evaluation results in a table format."""
    import re
    from collections import defaultdict
    from pathlib import Path

    logs_dir = Path("/home/elias/Thesis/src/openvla/experiments/logs")

    # Find the two relevant log files
    distilled_log = logs_dir / "EVAL-libero_spatial-openvla-2025_11_22-16_03_44.txt"
    vanilla_log = logs_dir / "EVAL-libero_spatial-openvla-2025_11_23-03_18_41.txt"

    def parse_log(log_path):
        """Parse evaluation log and return results by task."""
        tasks_results = defaultdict(lambda: {"successes": 0, "total": 0})

        if not log_path.exists():
            return tasks_results

        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        current_task = None
        for line in lines:
            if line.startswith("Task:"):
                current_task = line.replace("Task:", "").strip()

            if "Success:" in line:
                is_success = "Success: True" in line
                if current_task:
                    if is_success:
                        tasks_results[current_task]["successes"] += 1
                    tasks_results[current_task]["total"] += 1

        return tasks_results

    distilled_results = parse_log(distilled_log)
    vanilla_results = parse_log(vanilla_log)

    print("\n" + "="*100)
    print("EVALUATION RESULTS COMPARISON")
    print("="*100 + "\n")

    # Print table header
    print("┌" + "─" * 50 + "┬" + "─" * 22 + "┬" + "─" * 22 + "┐")
    print(f"│ {'Task':<48} │ {'Distilled':<20} │ {'Vanilla':<20} │")
    print("├" + "─" * 50 + "┼" + "─" * 22 + "┼" + "─" * 22 + "┤")

    all_tasks = sorted(set(list(distilled_results.keys()) + list(vanilla_results.keys())))

    total_distilled_success = 0
    total_distilled_episodes = 0
    total_vanilla_success = 0
    total_vanilla_episodes = 0

    for task in all_tasks:
        # Shorten task name for display
        task_display = task[:46] + ".." if len(task) > 48 else task

        d_counts = distilled_results.get(task, {"successes": 0, "total": 0})
        v_counts = vanilla_results.get(task, {"successes": 0, "total": 0})

        d_success = d_counts["successes"]
        d_total = d_counts["total"]
        v_success = v_counts["successes"]
        v_total = v_counts["total"]

        d_rate = (d_success / d_total * 100) if d_total > 0 else 0
        v_rate = (v_success / v_total * 100) if v_total > 0 else 0

        d_str = f"{d_success}/{d_total} ({d_rate:>5.1f}%)"
        v_str = f"{v_success}/{v_total} ({v_rate:>5.1f}%)"
        print(f"│ {task_display:<48} │ {d_str:<20} │ {v_str:<20} │")

        total_distilled_success += d_success
        total_distilled_episodes += d_total
        total_vanilla_success += v_success
        total_vanilla_episodes += v_total

    # Print totals
    total_d_rate = (total_distilled_success / total_distilled_episodes * 100) if total_distilled_episodes > 0 else 0
    total_v_rate = (total_vanilla_success / total_vanilla_episodes * 100) if total_vanilla_episodes > 0 else 0

    print("├" + "─" * 50 + "┼" + "─" * 22 + "┼" + "─" * 22 + "┤")
    total_d_str = f"{total_distilled_success}/{total_distilled_episodes} ({total_d_rate:>5.1f}%)"
    total_v_str = f"{total_vanilla_success}/{total_vanilla_episodes} ({total_v_rate:>5.1f}%)"
    print(f"│ {'TOTAL':<48} │ {total_d_str:<20} │ {total_v_str:<20} │")
    print("└" + "─" * 50 + "┴" + "─" * 22 + "┴" + "─" * 22 + "┘\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare rollout videos side-by-side")
    parser.add_argument("distilled_dir", help="Path to distilled model rollouts")
    parser.add_argument("vanilla_dir", help="Path to vanilla model rollouts")
    args = parser.parse_args()

    try:
        # Start the rollout viewer
        comparer = RolloutComparer(args.distilled_dir, args.vanilla_dir)
        comparer.run()
        print("\nExited successfully")
    except Exception as e:
        print(f"Error: {e}")
