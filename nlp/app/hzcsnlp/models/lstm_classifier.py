#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file lstm_classifier.py
# @brief
# @author QRS
# @version 1.0
# @date 2019-11-13 14:26:25

# ref: allennlp/models/basic_classifier.py

from typing import Dict, Optional

from overrides import overrides
import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy

@Model.register("lstm_classifier")
class LstmClassifier(Model):
    def __init__(self,
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            seq2vec_encoder: Seq2VecEncoder,
            seq2seq_encoder: Seq2SeqEncoder = None,
            dropout: float = None,
            num_labels: int = None,
            label_namespace: str = "labels",
            initializer: InitializerApplicator = InitializerApplicator(),
            regularizer: Optional[RegularizerApplicator] = None) -> None:

        super().__init__(vocab, regularizer)
        self._text_field_embedder = text_field_embedder

        if seq2seq_encoder:
            self._seq2seq_encoder = seq2seq_encoder
        else:
            self._seq2seq_encoder = None

        self._seq2vec_encoder = seq2vec_encoder
        self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None

        self._label_namespace = label_namespace

        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._metrics = {
                "accuracy": CategoricalAccuracy(),
                "accuracy3": CategoricalAccuracy(top_k=3),
                "accuracy5": CategoricalAccuracy(top_k=5),
        }
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self,
            tokens: Dict[str, torch.LongTensor],
            label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens).float()

        if self._seq2seq_encoder:
            embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

        embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        logits = self._classification_layer(embedded_text)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}

        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            for metric in self._metrics.values():
                metric(logits, label)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predictions = output_dict["probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = (self.vocab.get_index_to_token_vocabulary(self._label_namespace)
                    .get(label_idx, str(label_idx)))
            classes.append(label_str)
        output_dict["label"] = classes
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {name: metric.get_metric(reset) for
                             name, metric in self._metrics.items()}
        return metrics
