from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class OpenvlaLiberoSpatial(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation. Context view from fixed camera position.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Wrist camera RGB observation. Egocentric view from wrist-mounted camera.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc='Robot end-effector state, consists of [3x EEF position (x,y,z), '
                                '3x EEF orientation (euler angles or rotation), 2x gripper state (left, right finger positions)].',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, consists of [6x end-effector pose delta (3x position + 3x orientation), '
                            '1x gripper command (open/close)].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language instruction describing the task (e.g., "pick up the red block and place it in the drawer").'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'episode_index': tfds.features.Scalar(
                        dtype=np.int32,
                        doc='Unique episode identifier/index.'
                    ),
                    'task_index': tfds.features.Scalar(
                        dtype=np.int32,
                        doc='Task category index from LIBERO task suite.'
                    ),
                }),
            }))


    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='data/train_100eps/episode_*.npy'),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # Load raw data
            data = np.load(episode_path, allow_pickle=True).item()
            
            # Extract episode-level data
            actions = data['action']  # shape: (num_steps, 7)
            observations = data['observation']  # dict with arrays
            language_instruction = data['language_instruction']  # string
            episode_metadata = data['episode_metadata']  # dict
            
            # Compute language embedding once for the whole episode
            language_embedding = self._embed([language_instruction])[0].numpy()
            
            # Get number of steps
            num_steps = len(actions)
            
            # Assemble episode
            episode = []
            for i in range(num_steps):
                episode.append({
                    'observation': {
                        'image': observations['image'][i],
                        'wrist_image': observations['wrist_image'][i],
                        'state': observations['state'][i],
                    },
                    'action': actions[i],
                    'discount': 1.0,
                    'reward': float(i == (num_steps - 1)),
                    'is_first': i == 0,
                    'is_last': i == (num_steps - 1),
                    'is_terminal': i == (num_steps - 1),
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                })
            
            # Create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path,
                    'episode_index': episode_metadata.get('episode_index', 0),
                    'task_index': episode_metadata.get('task_index', 0),
                }
            }
            
            return episode_path, sample

        episode_paths = glob.glob(path)

        for sample in episode_paths:
            yield _parse_example(sample)

