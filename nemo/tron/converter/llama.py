from pathlib import Path
from typing import TYPE_CHECKING

import torch
import yaml

from nemo.collections.llm.gpt.model.base import torch_dtype_from_dict_config
from nemo.collections.llm.gpt.model.llama import (
    Llama31Config,
    _export_embedding,
    _export_head,
    _export_linear_fc1,
    _export_qkv,
)
from nemo.lightning import io
from nemo.lightning.io.state import _ModelState
from nemo.tron.container.utils.instantiate import instantiate
from nemo.tron.converter.common import get_full_mcore_state_dict
from nemo.tron.tokenizers.tokenizer import build_tokenizer
from nemo.utils import logging

if TYPE_CHECKING:
    from transformers import LlamaConfig as HFLlamaConfig
    from transformers import LlamaForCausalLM


class HFLlamaTronExporter:
    """Exporter to convert NeMo Llama models to Hugging Face format.

    This class handles the conversion process from a NeMo Llama model checkpoint
    to the Hugging Face Transformers format. It extracts the model weights and
    configuration from a NeMo checkpoint, maps them to the corresponding Hugging Face
    structure, and saves the result as a Hugging Face model.

    Args:
        input_path (Path): Path to the NeMo model checkpoint directory
        output_path (Path): Path where the converted Hugging Face model will be saved

    Example:
        ```python
        from pathlib import Path
        from nemo.tron.converter.llama import HFLlamaTronExporter

        # Define paths
        nemo_model_path = Path("/path/to/nemo/llama/model")
        hf_output_path = Path("/path/to/save/hf/model")

        # Initialize the exporter
        exporter = HFLlamaTronExporter(
            input_path=nemo_model_path,
            output_path=hf_output_path
        )

        # Perform the conversion
        output_dir = exporter.apply()
        print(f"Model converted and saved to: {output_dir}")

        # Load the converted model with Hugging Face
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(hf_output_path)
        tokenizer = AutoTokenizer.from_pretrained(hf_output_path)
        ```

    Notes:
        - The conversion process may require significant memory depending on model size
        - The exporter handles mapping between different weight naming conventions
        - Best used with NeMo Llama models trained with the NeMo framework
    """

    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = input_path
        self.output_path = output_path
        self.tokenizer = None
        self._config = None

    def init_hf_model(self, dtype=torch.bfloat16) -> "LlamaForCausalLM":
        """Initialize a new Hugging Face Llama model with the specified data type.

        Args:
            dtype (torch.dtype): The data type for the model parameters. Default: torch.bfloat16

        Returns:
            LlamaForCausalLM: An initialized Hugging Face Llama model with no weights loaded
        """
        from transformers import AutoModelForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights(True):
            return AutoModelForCausalLM.from_config(self.config, torch_dtype=dtype)

    def ckpt_load(self, path: Path) -> tuple[dict, dict]:
        """
        This function loads the state dict directly from a distributed checkpoint, and modify the state dict
        so that it is consistent with the key names you would get from loading the checkpoint into a model.
        This is a more memory-efficient method to obtain a state dict without initializing the nemo model.

        Args:
            path (Path): The path from which the model will be loaded.

        Returns
        -------
            Tuple[Dict, Dict]: The loaded state dict and the yaml config dict.
        """
        tron_yaml = path / "run_config.yaml"
        assert tron_yaml.exists()
        with open(tron_yaml, "r") as stream:
            config = yaml.safe_load(stream)
        config = config["model_config"]
        config = instantiate(config)
        self.tokenizer = build_tokenizer(instantiate(config["tokenizer_config"]))

        dist_ckpt_folder = path / "weights" if (path / "weights").exists() else path
        state_dict = {}
        state_dict = get_full_mcore_state_dict(dist_ckpt_folder)

        return state_dict, config

    def apply(self) -> Path:
        """Execute the conversion process from NeMo to Hugging Face format.

        This method:
        1. Loads the NeMo checkpoint
        2. Initializes a new Hugging Face model
        3. Converts and transfers the weights
        4. Saves the converted model to the output path

        Returns:
            Path: The path where the converted model is saved
        """
        logging.info("Loading Llama checkpoint. This may take a while...")
        source, source_config = self.ckpt_load(self)
        self._source_config = source_config
        logging.info("Llama checkpoint loaded.")
        target = self.init_hf_model(torch_dtype_from_dict_config(source_config))
        target = self.convert_state(source, target)

        target = target.cpu()
        if self.config.tie_word_embeddings:
            state_dict = target.state_dict()
            state_dict.pop("lm_head.weight")
            target.save_pretrained(self.output_path, state_dict=state_dict)
        else:
            target.save_pretrained(self.output_path)

        try:
            self.tokenizer.tokenizer.save_pretrained(self.output_path)
        except Exception:
            logging.warning("Failed to save tokenizer")

        return self.output_path

    def convert_state(self, source, target):
        """Convert the NeMo model state dictionary to the Hugging Face format.

        Args:
            source (dict): The state dictionary from the NeMo model
            target (LlamaForCausalLM): The target Hugging Face model

        Returns:
            LlamaForCausalLM: The target model with weights transferred from the source
        """
        mapping = {
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
        }
        transforms = [_export_qkv, _export_linear_fc1, _export_embedding]
        if not self.config.tie_word_embeddings:
            transforms.append(_export_head)

        _source = _ModelState(source)
        _source.config = self._source_config
        return io.apply_transforms(
            _source,
            target,
            mapping=mapping,
            transforms=transforms,
        )

    @property
    def config(self) -> "HFLlamaConfig":
        """Generate a Hugging Face Llama configuration from the NeMo model configuration.

        This property maps NeMo configuration parameters to their Hugging Face equivalents.

        Returns:
            HFLlamaConfig: A Hugging Face Llama configuration
        """
        if self._config is not None:
            return self._config

        source = self._source_config
        from transformers import LlamaConfig as HFLlamaConfig

        rope_scaling = None
        # For Llama 3.1 and Llama 3.2, rope_scaling is used and thus needed to parsed to the config
        if isinstance(source, Llama31Config):
            rope_scaling = {
                "factor": source.scale_factor,
                "low_freq_factor": source.low_freq_factor,
                "high_freq_factor": source.high_freq_factor,
                "original_max_position_embeddings": source.old_context_len,
                "rope_type": "llama3",
            }

        self._config = HFLlamaConfig(
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            max_position_embeddings=source.seq_length,
            initializer_range=source.init_method_std,
            rms_norm_eps=source.layernorm_epsilon,
            num_key_value_heads=source.num_query_groups,
            rope_theta=source.rotary_base,
            vocab_size=source.vocab_size,
            tie_word_embeddings=source.share_embeddings_and_output_weights,
            rope_scaling=rope_scaling,
            bos_token_id=self.tokenizer.bos_id if self.tokenizer else None,
            eos_token_id=self.tokenizer.eos_id if self.tokenizer else None,
        )
        return self._config
