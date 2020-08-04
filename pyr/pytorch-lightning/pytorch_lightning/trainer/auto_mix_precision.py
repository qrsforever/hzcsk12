from abc import ABC

from pytorch_lightning import _logger as log
from pytorch_lightning.utilities import rank_zero_warn, APEX_AVAILABLE, NATIVE_AMP_AVALAIBLE
from pytorch_lightning.utilities.distributed import rank_zero_debug


class TrainerAMPMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
    precision: int

    def init_amp(self):
        if NATIVE_AMP_AVALAIBLE:
            log.debug("`amp_level` has been deprecated since v0.7.4 (native amp does not require it)")

        assert self.precision in (16, 32), 'only 32 or 16 bit precision supported'

        if self.use_amp and NATIVE_AMP_AVALAIBLE:
            log.info('Using native 16bit precision.')
            return

        if self.use_amp and not APEX_AVAILABLE:  # pragma: no-cover
            raise ModuleNotFoundError(
                "You set `use_amp=True` but do not have apex installed."
                " Install apex first using this guide: https://github.com/NVIDIA/apex#linux"
                " and rerun with `use_amp=True`."
                " This run will NOT use 16 bit precision."
            )

        if self.use_amp:
            log.info('Using APEX 16bit precision.')

    @property
    def use_amp(self) -> bool:
        return self.precision == 16
