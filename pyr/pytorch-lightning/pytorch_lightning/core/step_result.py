from typing import Optional, Dict, Union, Sequence, Callable, MutableMapping, Any
from torch import Tensor
import torch
from copy import copy


class Result(Dict):

    def __init__(
            self,
            minimize: Optional[Tensor] = None,
            early_stop_on: Optional[Tensor] = None,
            checkpoint_on: Union[Tensor, bool, None] = None,
            hiddens: Optional[Tensor] = None,
    ):

        super().__init__()

        if early_stop_on is not None:
            self.early_stop_on = early_stop_on
        if checkpoint_on is not None and checkpoint_on:
            self.checkpoint_on = checkpoint_on
        if hiddens is not None:
            self.hiddens = hiddens
        if minimize is not None:
            err = 'Minimize can only be used in training_step, training_step_end, training_epoch_end'
            self._assert_grad_tensor_metric('minimize', minimize, err)
            self.minimize = minimize

        if minimize is not None and checkpoint_on is None:
            self.checkpoint_on = minimize.detach()

        self['meta'] = {
            '_internal': {
                '_reduce_on_epoch': False
            }
        }

    def __getattr__(self, key: str) -> Any:
        try:
            if key == 'callback_metrics':
                return self.get_callback_metrics()
            elif key == 'batch_log_metrics':
                return self.get_batch_log_metrics()
            elif key == 'batch_pbar_metrics':
                return self.get_batch_pbar_metrics()
            elif key == 'epoch_log_metrics':
                return self.get_epoch_log_metrics()
            elif key == 'epoch_pbar_metrics':
                return self.get_epoch_pbar_metrics()
            else:
                return self[key]
        except KeyError:
            return None

    def __setattr__(self, key: str, val: Union[Tensor, Any]):
        # ensure reserve keys are tensors and detached
        if key in {'hiddens', 'checkpoint_on', 'early_stop_on'}:
            self._assert_tensor_metric(key, val)
            if val is not None and isinstance(val, torch.Tensor):
                val = val.detach()

        # ensure anything else that is a tensor is detached
        elif isinstance(val, torch.Tensor) and key != 'minimize':
            val = val.detach()

        self[key] = val

    def _assert_tensor_metric(self, name: str, potential_metric: Union[bool, Tensor, None, Any]):
        if potential_metric is not None and not isinstance(potential_metric, bool):
            assert isinstance(potential_metric, Tensor), f'{name} must be a torch.Tensor'

    def _assert_grad_tensor_metric(self, name: str, x: Union[torch.Tensor, Any], additional_err: str = ''):
        if x is not None:
            assert isinstance(x, Tensor), f'{name} must be a torch.Tensor'
            m = f'{name} must have a computational graph.'

            if additional_err:
                m += f' {additional_err}'
            assert x.grad_fn is not None, m

    def log(
            self,
            name: str,
            value: Any,
            prog_bar: bool = False,
            logger: bool = True,
            on_step: bool = False,
            on_epoch: bool = True,
            reduce_fx: Callable = torch.mean,
            enable_graph: bool = False,
    ):
        # no metrics should be logged with graphs
        if not enable_graph and isinstance(value, torch.Tensor):
            value = value.detach()

        if 'meta' not in self:
            self.__setitem__('meta', {})

        # if user requests both step and epoch, then we split the metric in two automatically
        # one will be logged per step. the other per epoch
        if on_step and on_epoch:
            # set step version
            step_name = f'step_{name}'
            self.__set_meta(step_name, value, prog_bar, logger, on_step=True, on_epoch=False, reduce_fx=reduce_fx)
            self.__setitem__(step_name, value)

            # set epoch version
            epoch_name = f'epoch_{name}'
            self.__set_meta(epoch_name, value, prog_bar, logger, on_step=False, on_epoch=True, reduce_fx=reduce_fx)
            self.__setitem__(epoch_name, value)
        else:
            self.__set_meta(name, value, prog_bar, logger, on_step, on_epoch, reduce_fx)

            # set the value
            self.__setitem__(name, value)

    def __set_meta(
            self,
            name: str,
            value: Any,
            prog_bar: bool,
            logger: bool,
            on_step: bool,
            on_epoch: bool,
            reduce_fx: Callable,
    ):
        # set the meta for the item
        meta_value = value
        meta = dict(
            prog_bar=prog_bar,
            logger=logger,
            on_step=on_step,
            on_epoch=on_epoch,
            reduce_fx=reduce_fx,
            value=meta_value
        )

        self['meta'][name] = meta

        # track whether any input requires reduction on epoch end
        _internal = self['meta']['_internal']
        _internal['_reduce_on_epoch'] = max(_internal['_reduce_on_epoch'], on_epoch)

    def get_callback_metrics(self) -> dict:
        result = {
            'early_stop_on': self.early_stop_on,
            'checkpoint_on': self.checkpoint_on
        }

        return result

    def get_batch_log_metrics(self) -> dict:
        """
        Gets the metrics to log at the end of the batch step
        """
        result = {}

        meta = self['meta']
        for k, options in meta.items():
            if k == '_internal':
                continue
            if options['logger'] and options['on_step']:
                result[k] = self[k]
        return result

    def get_epoch_log_metrics(self) -> dict:
        """
        Gets the metrics to log at the end of the batch step
        """
        result = {}

        meta = self['meta']
        for k, options in meta.items():
            if k == '_internal':
                continue
            if options['logger'] and options['on_epoch']:
                result[k] = self[k]
        return result

    def get_epoch_pbar_metrics(self):
        """
        Gets the metrics to log at the end of the batch step
        """
        result = {}

        meta = self['meta']
        for k, options in meta.items():
            if k == '_internal':
                continue
            if options['prog_bar'] and options['on_epoch']:
                result[k] = self[k]
        return result

    def get_batch_pbar_metrics(self):
        """
        Gets the metrics to log at the end of the batch step
        """
        result = {}

        meta = self['meta']
        for k, options in meta.items():
            if k == '_internal':
                continue
            if options['prog_bar'] and options['on_step']:
                result[k] = self[k]
        return result

    def detach(self):
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                self.__setitem__(k, v.detach())

    def __repr__(self):
        self_copy = self.copy()

        if 'meta' in self_copy:
            del self_copy['meta']

        return str(self_copy)

    def __str__(self):
        copy = self.copy()
        del copy['meta']

        return str(copy)

    def __copy__(self):
        newone = type(self)()
        for k, v in self.items():
            newone[k] = copy(v)
        return newone

    @classmethod
    def gather(cls, outputs):
        meta = outputs[0].get('meta')
        result = cls()
        result = recursive_gather(outputs, result)
        recursive_stack(result)

        if meta:
            result['meta'] = meta
        return result

    @classmethod
    def reduce_on_epoch_end(cls, outputs):
        meta = outputs[0]['meta']
        result = cls()
        result = recursive_gather(outputs, result)
        recursive_stack(result)

        for k, option in meta.items():
            if k == '_internal':
                continue

            if option['on_epoch']:
                fx = option['reduce_fx']
                result[k] = fx(result[k])

        result['meta'] = meta
        return result

    @property
    def should_reduce_on_epoch_end(self) -> bool:
        return self['meta']['_internal']['_reduce_on_epoch']


def recursive_gather(outputs: Sequence[dict], result: Optional[MutableMapping] = None) -> Optional[MutableMapping]:
    for out in outputs:
        if 'meta' in out:
            del out['meta']

        for k, v in out.items():
            if isinstance(v, dict):
                v = recursive_gather([v], result)

            if k not in result:
                result[k] = []

            result[k].append(v)

    return result


def recursive_stack(result: MutableMapping):
    for k, v in result.items():
        if isinstance(v, dict):
            recursive_stack(v)

        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            v = torch.stack(v)
            result[k] = v


class TrainResult(Result):

    def __init__(
            self,
            minimize: Optional[Tensor] = None,
            early_stop_on: Tensor = None,
            checkpoint_on: Union[Tensor, bool] = None,
            hiddens: Optional[Tensor] = None,
    ):
        """
        Used in train loop to auto-log to a logger or progress bar without needing to define
        a train_step_end or train_epoch_end method

        Example::

            def training_step(self, batch, batch_idx):
                loss = ...
                result = pl.TrainResult(loss)
                result.log('train_loss', loss)
                return result

            # without val/test loop can model checkpoint or early stop
            def training_step(self, batch, batch_idx):
                loss = ...
                result = pl.TrainResult(loss, early_stop_on=loss, checkpoint_on=loss)
                result.log('train_loss', loss)
                return result

        Args:
            early_stop_on:
            checkpoint_on:
            hiddens:
        """

        super().__init__(minimize, early_stop_on, checkpoint_on, hiddens)

    def log(
            self,
            name,
            value,
            prog_bar: bool = False,
            logger: bool = True,
            on_step: bool = True,
            on_epoch: bool = False,
            reduce_fx: Callable = torch.mean,
            enable_graph: bool = False,
    ):
        """
        Log a key, value

        Example::

            result.log('train_loss', loss)

            # defaults used
            result.log(
                name,
                value,
                on_step=True,
                on_epoch=False,
                logger=True,
                prog_bar=False,
                reduce_fx=torch.mean,
                enable_graph=False
            )


        Args:
            name: key name
            value: value name
            prog_bar: if True logs to the progress base
            logger: if True logs to the logger
            on_step: if True logs the output of validation_step or test_step
            on_epoch: if True, logs the output of the training loop aggregated
            reduce_fx: Torch.mean by default
            enable_graph: if True, will not auto detach the graph
        """
        super().log(name, value, prog_bar, logger, on_step, on_epoch, reduce_fx, enable_graph)

    def log_dict(
            self,
            dictionary: dict,
            prog_bar: bool = False,
            logger: bool = True,
            on_step: bool = False,
            on_epoch: bool = True,
            reduce_fx: Callable = torch.mean,
            enable_graph: bool = False,
    ):
        """
        Log a dictonary of values at once

        Example::

            values = {'loss': loss, 'acc': acc, ..., 'metric_n': metric_n}
            result.log_dict(values)

        Args:
            dictionary:
            prog_bar:
            logger:
            on_step:
            on_epoch:
            reduce_fx:
            enable_graph:
        """
        for k, v in dictionary.items():
            self.log(k, v, prog_bar, logger, on_step, on_epoch, reduce_fx, enable_graph)


class EvalResult(Result):

    def __init__(
            self,
            early_stop_on: Optional[Tensor] = None,
            checkpoint_on: Optional[Tensor] = None,
            hiddens: Optional[Tensor] = None,
    ):
        """
        Used in val/train loop to auto-log to a logger or progress bar without needing to define
        a _step_end or _epoch_end method

        Example::

            def validation_step(self, batch, batch_idx):
                loss = ...
                result = EvalResult()
                result.log('val_loss', loss)
                return result

            def test_step(self, batch, batch_idx):
                loss = ...
                result = EvalResult()
                result.log('val_loss', loss)
                return result

        Args:
            early_stop_on:
            checkpoint_on:
            hiddens:
        """

        super().__init__(None, early_stop_on, checkpoint_on, hiddens)

    def log(
            self,
            name,
            value,
            prog_bar: bool = False,
            logger: bool = True,
            on_step: bool = False,
            on_epoch: bool = True,
            reduce_fx: Callable = torch.mean,
            enable_graph: bool = False,
    ):
        """
        Log a key, value

        Example::

            result.log('val_loss', loss)

            # defaults used
            result.log(
                name,
                value,
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=False,
                reduce_fx=torch.mean
            )


        Args:
            name: key name
            value: value name
            prog_bar: if True logs to the progress base
            logger: if True logs to the logger
            on_step: if True logs the output of validation_step or test_step
            on_epoch: if True, logs the output of the validation loop or test loop aggregated
            reduce_fx: Torch.mean by default
            enable_graph: if True, will not auto detach the graph :
        """
        super().log(name, value, prog_bar, logger, on_step, on_epoch, reduce_fx, enable_graph)

    def log_dict(
            self,
            dictionary: dict,
            prog_bar: bool = False,
            logger: bool = True,
            on_step: bool = False,
            on_epoch: bool = True,
            reduce_fx: Callable = torch.mean,
            enable_graph: bool = False,
    ):
        """
        Log a dictonary of values at once

        Example::

            values = {'loss': loss, 'acc': acc, ..., 'metric_n': metric_n}
            result.log_dict(values)

        Args:
            dictionary:
            prog_bar:
            logger:
            on_step:
            on_epoch:
            reduce_fx:
            enable_graph:
        """
        for k, v in dictionary.items():
            self.log(k, v, prog_bar, logger, on_step, on_epoch, reduce_fx, enable_graph)

    def get_callback_metrics(self) -> dict:
        result = {
            'val_early_stop_on': self.early_stop_on,
            'val_checkpoint_on': self.checkpoint_on
        }

        return result
