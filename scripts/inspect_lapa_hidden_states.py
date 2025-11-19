"""
inspect_lapa_hidden_states.py

Read the extracted LAPA hidden states HDF5 file and display hidden state shapes
for a given episode.

Usage:
python inspect_lapa_hidden_states.py \
    --h5_file /path/to/lapa_hidden_states.h5 \
    --episode_idx 0
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from io import BytesIO
import subprocess

try:
    import h5py
except ImportError:
    print("❌ h5py not installed. Install with: pip install h5py")
    exit(1)

try:
    from sklearn.decomposition import PCA
except ImportError:
    print("❌ scikit-learn not installed. Install with: pip install scikit-learn")
    exit(1)


def extract_frame_from_video(video_path, frame_idx):
    """Extract a frame from video at given index using ffmpeg."""
    try:
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vf', f'select=eq(n\\,{frame_idx})',
            '-vframes', '1',
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-'
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=5)
        if result.returncode == 0:
            frame = Image.open(BytesIO(result.stdout)).convert('RGB')
            return frame
    except Exception:
        pass
    return None


def find_video_for_episode(dataset_dir, episode_idx):
    """Find video file for episode in dataset structure."""
    dataset_dir = Path(dataset_dir)
    videos_dir = dataset_dir / 'videos'

    if videos_dir.exists():
        for chunk_dir in sorted(videos_dir.glob('chunk-*')):
            image_dir = chunk_dir / 'observation.images.image'
            if image_dir.exists():
                potential_video = image_dir / f'episode_{episode_idx:06d}.mp4'
                if potential_video.exists():
                    return potential_video
    return None


def plot_pca_3d_per_frame(h5_file, episode_idx, output_dir=None, dataset_dir=None, vision_only=False):
    """
    Plot 3D PCA of hidden states with interactive frame navigation.

    Arrow Keys:
        LEFT: Previous frame
        RIGHT: Next frame
        ESC: Close

    Args:
        h5_file: Path to lapa_hidden_states.h5
        episode_idx: Episode index to plot
        output_dir: Directory to save the plot (optional)
        dataset_dir: Path to dataset for showing video frames
        vision_only: If True, only plot first 257 vision tokens; if False, plot full sequence
    """
    h5_file = Path(h5_file)

    if not h5_file.exists():
        print(f"❌ File not found: {h5_file}")
        return

    with h5py.File(h5_file, 'r') as f:
        # Find all samples for this episode
        episode_indices = f['episode_indices'][:]
        matching_indices = np.where(episode_indices == episode_idx)[0]

        if len(matching_indices) == 0:
            print(f"❌ No samples found for episode {episode_idx}")
            return

        print(f"\n🎨 Plotting 3D PCA for episode {episode_idx}...")

        # Collect all hidden states and frame indices
        all_hidden_states = []
        all_frame_indices = []
        frame_to_indices = {}  # Map frame_idx to data indices

        for sample_idx in matching_indices:
            hidden_state_flat = f['hidden_states'][sample_idx]
            seq_len = f['seq_lengths'][sample_idx]
            frame_idx = f['frame_indices'][sample_idx]

            hidden_state_2d = hidden_state_flat.reshape(seq_len, 4096)
            all_hidden_states.append(hidden_state_2d)

            # Repeat frame index for each token in this frame's sequence
            all_frame_indices.extend([frame_idx] * seq_len)

            if frame_idx not in frame_to_indices:
                frame_to_indices[frame_idx] = []
            frame_to_indices[frame_idx].append(len(all_frame_indices) - seq_len)

        # Concatenate all hidden states
        all_hidden_states_concat = np.vstack(all_hidden_states)
        all_frame_indices = np.array(all_frame_indices)

        print(f"   - Total vectors: {len(all_hidden_states_concat)}")
        print(f"   - Frames: {len(matching_indices)}")

        # Filter to vision tokens if requested
        if vision_only:
            # Vision tokens come after text tokens in the sequence
            # Structure: [text_tokens (128)] + [vision_tokens (~256-259)] + [continuation (~3-4)]
            # Total sequence length: ~391 tokens
            # Source: extract_hidden_states.py lines 217-219, 229-233

            # Text is padded to max_length=128
            # Vision tokens are VQGAN-encoded image tokens (approximately 256-259)
            # Continuation is "</vision> <delta>" (3-4 tokens)

            vision_start_pos = 128  # Skip text, start at vision tokens
            vision_end_pos = 391    # End of vision tokens (before continuation)

            vision_mask = []
            current_frame = None
            frame_token_idx = 0

            for i, frame_idx_val in enumerate(all_frame_indices):
                if current_frame != frame_idx_val:
                    current_frame = frame_idx_val
                    frame_token_idx = 0

                # Keep tokens in vision range [128:391]
                if vision_start_pos <= frame_token_idx < vision_end_pos:
                    vision_mask.append(True)
                else:
                    vision_mask.append(False)
                frame_token_idx += 1

            vision_mask = np.array(vision_mask)
            all_hidden_states_concat = all_hidden_states_concat[vision_mask]
            all_frame_indices = all_frame_indices[vision_mask]
            print(f"   - Filtered to vision tokens (positions 128-391): {len(all_hidden_states_concat)} vectors")

        # Compute PCA
        pca_label = "vision tokens" if vision_only else "full sequence"
        print(f"   - Computing 3D PCA ({pca_label})...")
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(all_hidden_states_concat)

        print(f"   - Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

        # Get unique frames
        unique_frames = np.unique(all_frame_indices)
        n_frames = len(unique_frames)
        cmap = cm.get_cmap('tab20' if n_frames <= 20 else 'hsv')
        norm = Normalize(vmin=unique_frames.min(), vmax=unique_frames.max())

        # Find video for frame extraction
        video_path = None
        if dataset_dir:
            video_path = find_video_for_episode(dataset_dir, episode_idx)

        # State for interactive navigation
        state = {'current_frame_idx': 0, 'fig': None, 'ax_3d': None, 'ax_frame': None, 'cached_frames': {}}

        def update_plot():
            """Redraw plot with current frame highlighted."""
            if state['ax_3d'] is not None:
                state['ax_3d'].clear()
            else:
                state['fig'] = plt.figure(figsize=(18, 8))
                state['ax_3d'] = state['fig'].add_subplot(121, projection='3d')
                state['ax_frame'] = state['fig'].add_subplot(122)

            ax_3d = state['ax_3d']
            ax_frame = state['ax_frame']
            current_frame = unique_frames[state['current_frame_idx']]

            # Plot 3D PCA with varying alpha
            for i, frame_idx in enumerate(unique_frames):
                mask = all_frame_indices == frame_idx
                color = cmap(norm(frame_idx))
                alpha = 0.9 if frame_idx == current_frame else 0.2
                ax_3d.scatter(
                    pca_result[mask, 0],
                    pca_result[mask, 1],
                    pca_result[mask, 2],
                    c=[color],
                    label=f'Frame {frame_idx}' if frame_idx == current_frame else '',
                    s=80 if frame_idx == current_frame else 30,
                    alpha=alpha,
                    edgecolors='gold' if frame_idx == current_frame else 'black',
                    linewidth=2.0 if frame_idx == current_frame else 0.3
                )

            ax_3d.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=10)
            ax_3d.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=10)
            ax_3d.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})', fontsize=10)

            current_frame = unique_frames[state['current_frame_idx']]
            mode_label = " (Vision Only)" if vision_only else " (Full Sequence)"
            ax_3d.set_title(
                f'3D PCA{mode_label} - Episode {episode_idx} | Frame {current_frame} ({state["current_frame_idx"] + 1}/{len(unique_frames)})',
                fontsize=12, fontweight='bold'
            )
            ax_3d.legend(loc='upper left', fontsize=8)

            # Display video frame
            if video_path:
                if current_frame not in state['cached_frames']:
                    frame_img = extract_frame_from_video(video_path, current_frame)
                    state['cached_frames'][current_frame] = frame_img
                else:
                    frame_img = state['cached_frames'][current_frame]

                if frame_img:
                    ax_frame.imshow(frame_img)
                    ax_frame.set_title(f'Frame {current_frame} from Video', fontsize=12, fontweight='bold')
                    ax_frame.axis('off')
                else:
                    ax_frame.text(0.5, 0.5, f'Could not extract\nframe {current_frame}',
                                 ha='center', va='center', fontsize=12, transform=ax_frame.transAxes)
                    ax_frame.axis('off')
            else:
                ax_frame.text(0.5, 0.5, 'No dataset_dir provided\n(use --dataset_dir)',
                             ha='center', va='center', fontsize=11, transform=ax_frame.transAxes)
                ax_frame.axis('off')

            plt.tight_layout()
            state['fig'].canvas.draw_idle()

        def on_key(event):
            """Handle keyboard navigation."""
            if event.key == 'left':
                state['current_frame_idx'] = max(0, state['current_frame_idx'] - 1)
                print(f"← Frame {unique_frames[state['current_frame_idx']]}")
                update_plot()
            elif event.key == 'right':
                state['current_frame_idx'] = min(len(unique_frames) - 1, state['current_frame_idx'] + 1)
                print(f"→ Frame {unique_frames[state['current_frame_idx']]}")
                update_plot()
            elif event.key == 'escape':
                plt.close(state['fig'])

        # Create initial plot
        state['fig'] = plt.figure(figsize=(18, 8))
        state['ax_3d'] = state['fig'].add_subplot(121, projection='3d')
        state['ax_frame'] = state['fig'].add_subplot(122)

        state['fig'].canvas.mpl_connect('key_press_event', on_key)

        print(f"\n⌨️  Controls: ← LEFT arrow (previous frame) | RIGHT arrow → (next frame) | ESC (close)")

        update_plot()

        # Save or show
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / f"pca_3d_episode_{episode_idx}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Plot saved: {save_path}")
        else:
            plt.show()


def plot_pca_per_frame(h5_file, episode_idx, output_dir=None, dataset_dir=None, vision_only=False):
    """
    Plot 2D PCA of hidden states with interactive frame navigation.

    Arrow Keys:
        LEFT: Previous frame
        RIGHT: Next frame
        ESC: Close

    Args:
        h5_file: Path to lapa_hidden_states.h5
        episode_idx: Episode index to plot
        output_dir: Directory to save the plot (optional)
        dataset_dir: Path to dataset for showing video frames
        vision_only: If True, only plot first 257 vision tokens; if False, plot full sequence
    """
    h5_file = Path(h5_file)

    if not h5_file.exists():
        print(f"❌ File not found: {h5_file}")
        return

    with h5py.File(h5_file, 'r') as f:
        # Find all samples for this episode
        episode_indices = f['episode_indices'][:]
        matching_indices = np.where(episode_indices == episode_idx)[0]

        if len(matching_indices) == 0:
            print(f"❌ No samples found for episode {episode_idx}")
            return

        print(f"\n🎨 Plotting 2D PCA for episode {episode_idx}...")

        # Collect all hidden states and frame indices
        all_hidden_states = []
        all_frame_indices = []

        for sample_idx in matching_indices:
            hidden_state_flat = f['hidden_states'][sample_idx]
            seq_len = f['seq_lengths'][sample_idx]
            frame_idx = f['frame_indices'][sample_idx]

            hidden_state_2d = hidden_state_flat.reshape(seq_len, 4096)
            all_hidden_states.append(hidden_state_2d)

            # Repeat frame index for each token in this frame's sequence
            all_frame_indices.extend([frame_idx] * seq_len)

        # Concatenate all hidden states
        all_hidden_states_concat = np.vstack(all_hidden_states)
        all_frame_indices = np.array(all_frame_indices)

        print(f"   - Total vectors: {len(all_hidden_states_concat)}")
        print(f"   - Frames: {len(matching_indices)}")

        # Filter to vision tokens if requested
        if vision_only:
            # Vision tokens come after text tokens in the sequence
            # Structure: [text_tokens (128)] + [vision_tokens (~256-259)] + [continuation (~3-4)]
            # Total sequence length: ~391 tokens
            # Source: extract_hidden_states.py lines 217-219, 229-233

            # Text is padded to max_length=128
            # Vision tokens are VQGAN-encoded image tokens (approximately 256-259)
            # Continuation is "</vision> <delta>" (3-4 tokens)

            vision_start_pos = 128  # Skip text, start at vision tokens
            vision_end_pos = 391    # End of vision tokens (before continuation)

            vision_mask = []
            current_frame = None
            frame_token_idx = 0

            for i, frame_idx_val in enumerate(all_frame_indices):
                if current_frame != frame_idx_val:
                    current_frame = frame_idx_val
                    frame_token_idx = 0

                # Keep tokens in vision range [128:391]
                if vision_start_pos <= frame_token_idx < vision_end_pos:
                    vision_mask.append(True)
                else:
                    vision_mask.append(False)
                frame_token_idx += 1

            vision_mask = np.array(vision_mask)
            all_hidden_states_concat = all_hidden_states_concat[vision_mask]
            all_frame_indices = all_frame_indices[vision_mask]
            print(f"   - Filtered to vision tokens (positions 128-391): {len(all_hidden_states_concat)} vectors")

        # Compute PCA
        pca_label = "vision tokens" if vision_only else "full sequence"
        print(f"   - Computing 2D PCA ({pca_label})...")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(all_hidden_states_concat)

        print(f"   - Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

        # Get unique frames
        unique_frames = np.unique(all_frame_indices)
        n_frames = len(unique_frames)
        cmap = cm.get_cmap('tab20' if n_frames <= 20 else 'hsv')
        norm = Normalize(vmin=unique_frames.min(), vmax=unique_frames.max())

        # Find video for frame extraction
        video_path = None
        if dataset_dir:
            video_path = find_video_for_episode(dataset_dir, episode_idx)

        # State for interactive navigation
        state = {'current_frame_idx': 0, 'fig': None, 'ax_2d': None, 'ax_frame': None, 'cached_frames': {}}

        def update_plot():
            """Redraw plot with current frame highlighted."""
            if state['ax_2d'] is not None:
                state['ax_2d'].clear()
            else:
                state['fig'] = plt.figure(figsize=(16, 7))
                state['ax_2d'] = state['fig'].add_subplot(121)
                state['ax_frame'] = state['fig'].add_subplot(122)

            ax_2d = state['ax_2d']
            ax_frame = state['ax_frame']
            current_frame = unique_frames[state['current_frame_idx']]

            # Plot all frames with varying alpha
            for frame_idx in unique_frames:
                mask = all_frame_indices == frame_idx
                color = cmap(norm(frame_idx))
                alpha = 0.9 if frame_idx == current_frame else 0.2
                ax_2d.scatter(
                    pca_result[mask, 0],
                    pca_result[mask, 1],
                    c=[color],
                    label=f'Frame {frame_idx}' if frame_idx == current_frame else '',
                    s=80 if frame_idx == current_frame else 30,
                    alpha=alpha,
                    edgecolors='gold' if frame_idx == current_frame else 'black',
                    linewidth=2.0 if frame_idx == current_frame else 0.3
                )

            ax_2d.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
            ax_2d.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)

            current_frame = unique_frames[state['current_frame_idx']]
            mode_label = " (Vision Only)" if vision_only else " (Full Sequence)"
            ax_2d.set_title(
                f'2D PCA{mode_label} - Episode {episode_idx} | Frame {current_frame} ({state["current_frame_idx"] + 1}/{len(unique_frames)})',
                fontsize=12, fontweight='bold'
            )
            ax_2d.legend(loc='upper left', fontsize=9)
            ax_2d.grid(True, alpha=0.3)

            # Display video frame
            if video_path:
                if current_frame not in state['cached_frames']:
                    frame_img = extract_frame_from_video(video_path, current_frame)
                    state['cached_frames'][current_frame] = frame_img
                else:
                    frame_img = state['cached_frames'][current_frame]

                if frame_img:
                    ax_frame.imshow(frame_img)
                    ax_frame.set_title(f'Frame {current_frame} from Video', fontsize=12, fontweight='bold')
                    ax_frame.axis('off')
                else:
                    ax_frame.text(0.5, 0.5, f'Could not extract\nframe {current_frame}',
                                 ha='center', va='center', fontsize=12, transform=ax_frame.transAxes)
                    ax_frame.axis('off')
            else:
                ax_frame.text(0.5, 0.5, 'No dataset_dir provided\n(use --dataset_dir)',
                             ha='center', va='center', fontsize=11, transform=ax_frame.transAxes)
                ax_frame.axis('off')

            plt.tight_layout()
            state['fig'].canvas.draw_idle()

        def on_key(event):
            """Handle keyboard navigation."""
            if event.key == 'left':
                state['current_frame_idx'] = max(0, state['current_frame_idx'] - 1)
                print(f"← Frame {unique_frames[state['current_frame_idx']]}")
                update_plot()
            elif event.key == 'right':
                state['current_frame_idx'] = min(len(unique_frames) - 1, state['current_frame_idx'] + 1)
                print(f"→ Frame {unique_frames[state['current_frame_idx']]}")
                update_plot()
            elif event.key == 'escape':
                plt.close(state['fig'])

        # Create initial plot
        state['fig'] = plt.figure(figsize=(16, 7))
        state['ax_2d'] = state['fig'].add_subplot(121)
        state['ax_frame'] = state['fig'].add_subplot(122)

        state['fig'].canvas.mpl_connect('key_press_event', on_key)

        print(f"\n⌨️  Controls: ← LEFT arrow (previous frame) | RIGHT arrow → (next frame) | ESC (close)")

        update_plot()

        # Save or show
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / f"pca_episode_{episode_idx}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Plot saved: {save_path}")
        else:
            plt.show()


def inspect_episode(h5_file, episode_idx):
    """
    Read HDF5 file and display hidden state shapes for a specific episode.

    Args:
        h5_file: Path to lapa_hidden_states.h5
        episode_idx: Episode index to inspect
    """
    h5_file = Path(h5_file)

    if not h5_file.exists():
        print(f"❌ File not found: {h5_file}")
        return

    print(f"\n{'='*70}")
    print(f"Reading: {h5_file}")
    print(f"{'='*70}")

    with h5py.File(h5_file, 'r') as f:
        print(f"\n📊 Dataset Info:")
        print(f"  - Total samples: {len(f['hidden_states'])}")
        print(f"  - Keys: {list(f.keys())}")

        # Find all samples for this episode
        episode_indices = f['episode_indices'][:]
        unique_episodes = np.unique(episode_indices)
        print(f"  - Episodes with data: {len(unique_episodes)}")
        print(f"  - Episode range: {unique_episodes.min()} to {unique_episodes.max()}")
        print(f"  - Missing episodes: {432 - len(unique_episodes)} (likely failed or had no video)")

        matching_indices = np.where(episode_indices == episode_idx)[0]

        if len(matching_indices) == 0:
            print(f"\n❌ No samples found for episode {episode_idx}")
            print(f"   Available episodes: {np.unique(episode_indices).tolist()}")
            return

        print(f"\n✅ Found {len(matching_indices)} samples for episode {episode_idx}")
        print(f"\n{'='*70}")
        print(f"Episode {episode_idx} - Hidden State Shapes")
        print(f"{'='*70}\n")

        # Print details for each sample in this episode
        print(f"{'Index':<8} {'Frame':<8} {'Seq Len':<10} {'Shape':<20} {'Task'}")
        print(f"{'-'*70}")

        for sample_idx in matching_indices:
            # Read metadata
            frame_idx = f['frame_indices'][sample_idx]
            seq_len = f['seq_lengths'][sample_idx]
            task_desc = f['task_descriptions'][sample_idx]

            # Reconstruct 2D hidden state
            hidden_state_flat = f['hidden_states'][sample_idx]
            hidden_state_2d = hidden_state_flat.reshape(seq_len, 4096)

            # Decode task description if bytes
            if isinstance(task_desc, bytes):
                task_desc = task_desc.decode('utf-8')

            # Truncate task description for display
            task_desc_short = task_desc[:30] + "..." if len(task_desc) > 30 else task_desc

            shape_str = str(hidden_state_2d.shape)
            print(f"{sample_idx:<8} {frame_idx:<8} {seq_len:<10} {shape_str:<20} {task_desc_short}")

        # Statistics
        print(f"\n{'='*70}")
        print(f"Statistics for Episode {episode_idx}:")
        print(f"{'='*70}")

        seq_lens = f['seq_lengths'][matching_indices]
        print(f"  - Sequence length range: {seq_lens.min()} to {seq_lens.max()}")
        print(f"  - Average sequence length: {seq_lens.mean():.1f}")
        print(f"  - Hidden dimension: 4096")

        # Concatenate all hidden states for this episode and show stats
        all_hidden_states = []
        for sample_idx in matching_indices:
            hidden_state_flat = f['hidden_states'][sample_idx]
            seq_len = f['seq_lengths'][sample_idx]
            hidden_state_2d = hidden_state_flat.reshape(seq_len, 4096)
            all_hidden_states.append(hidden_state_2d)

        all_hidden_states_concat = np.vstack(all_hidden_states)
        print(f"\n  - Total vectors in episode: {len(all_hidden_states_concat)}")
        print(f"  - Min value: {all_hidden_states_concat.min():.6f}")
        print(f"  - Max value: {all_hidden_states_concat.max():.6f}")
        print(f"  - Mean value: {all_hidden_states_concat.mean():.6f}")
        print(f"  - Std value: {all_hidden_states_concat.std():.6f}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect LAPA hidden states HDF5 file"
    )
    parser.add_argument(
        "--h5_file",
        type=str,
        required=True,
        help="Path to lapa_hidden_states.h5"
    )
    parser.add_argument(
        "--episode_idx",
        type=int,
        default=0,
        help="Episode index to inspect (default: 0)"
    )
    parser.add_argument(
        "--plot_pca",
        action="store_true",
        help="Plot 2D PCA of hidden states with different colors per frame"
    )
    parser.add_argument(
        "--plot_pca_3d",
        action="store_true",
        help="Plot 3D PCA of hidden states with different colors per frame"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save PCA plot (if not provided, displays in window)"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Path to libero_spatial_noops dataset (shows video frames)"
    )
    parser.add_argument(
        "--vision_only",
        action="store_true",
        help="Plot only vision tokens (first 257); default is full sequence"
    )

    args = parser.parse_args()

    if args.plot_pca_3d:
        plot_pca_3d_per_frame(args.h5_file, args.episode_idx, args.output_dir, args.dataset_dir, args.vision_only)
    elif args.plot_pca:
        plot_pca_per_frame(args.h5_file, args.episode_idx, args.output_dir, args.dataset_dir, args.vision_only)
    else:
        inspect_episode(args.h5_file, args.episode_idx)
