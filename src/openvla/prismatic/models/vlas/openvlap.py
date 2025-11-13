"""
openvla.py

PyTorch Module defining OpenVLA as a lightweight wrapper around a PrismaticVLM; defines custom logic around
discretizing actions with the ActionTokenizer.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import LlamaTokenizerFast

from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.vla.action_tokenizer import ActionTokenizer

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class OpenVLAP(PrismaticVLM):
    def __init__(
        self,
        *args,
        action_tokenizer: ActionTokenizer,
        action_dim: int,
        distill_projection_dim: int = 4,
        distill_projection_hidden_dim: int = 64,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.action_dim = action_dim
        self.action_tokenizer = action_tokenizer

        # Optional projection layer for distillation
        # Projects from action_dim (7) to teacher's latent dim (e.g., 4)
        self.distill_projection = torch.nn.Sequential(
            torch.nn.Linear(action_dim, distill_projection_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(distill_projection_hidden_dim, distill_projection_dim),
            torch.nn.Tanh(),
        )
        self.distill_norm = torch.nn.LayerNorm(
            distill_projection_dim, 
            elementwise_affine=True,  # learnable γ, β
            eps=1e-6
        )

    def predict_action(
        self, image: Image, instruction: str, **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action (de-tokenizes).

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.

        @return Normalized action vector (np.ndarray if return_projected=False, torch.Tensor if return_projected=True)
        """
        image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer

        # Build VLA Prompt
        prompt_builder = self.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        if isinstance(tokenizer, LlamaTokenizerFast):
            # If the special empty token ('') does not already appear after the colon (':') token in the prompt
            # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
                )
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        # Preprocess Image
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            # fmt: off
            generated_ids = super(PrismaticVLM, self).generate(
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, 3, res, res] or Dict[str, ...]
                max_new_tokens=self.get_action_dim(),
                **kwargs
            )
            # fmt: on

        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim() :]
        normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids.cpu().numpy())

        action_tensor = torch.from_numpy(normalized_actions).to(self.device).float()
        projected_actions = self.distill_projection(action_tensor)
        projected_actions = self.distill_norm(projected_actions)
        
        return projected_actions


    def get_action_dim(self) -> int:
        return self.action_dim
