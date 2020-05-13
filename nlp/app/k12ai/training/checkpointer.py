#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file checkpointer.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-05-13 14:47

from typing import Union, Dict, Any, Tuple
import os, shutil
import torch
from allennlp.training.checkpointer import Checkpointer as AllennlpCheckpointer


class Checkpointer(AllennlpCheckpointer):
    def __init__(
        self,
        serialization_dir: str = None,
        keep_serialized_model_every_num_seconds: int = None,
        num_serialized_models_to_keep: int = 20,
    ) -> None:
        super().__init__(serialization_dir,
                keep_serialized_model_every_num_seconds, num_serialized_models_to_keep)
        if serialization_dir is not None:
            self._model_state_path = os.path.join(
                self._serialization_dir, "model_state_latest.th"
            )
            self._training_state_path = os.path.join(
                self._serialization_dir, "training_state_latest.th"
            )
            self._model_best_path = os.path.join(
                self._serialization_dir, "best.th"
            )

    def save_checkpoint(
        self,
        epoch: Union[int, str],
        model_state: Dict[str, Any],
        training_states: Dict[str, Any],
        is_best_so_far: bool,
    ) -> None:
        if self._serialization_dir is not None:
            torch.save(model_state, self._model_state_path)
            torch.save({**training_states, "epoch": epoch}, self._training_state_path)
            if is_best_so_far:
                shutil.copyfile(self._model_state_path, self._model_best_path)

    def find_latest_checkpoint(self) -> Tuple[str, str]:
        if os.path.exists(self._model_state_path) and \
                os.path.exists(self._training_state_path):
            return (self._model_state_path, self._training_state_path)
        return None
