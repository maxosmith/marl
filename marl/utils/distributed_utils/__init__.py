# pylint: disable=unused-import

from marl.utils.distributed_utils.put_to_devices_iterable import (
    device_put, multi_device_put)
from marl.utils.distributed_utils.utils import (get_from_first_device,
                                                replicate_on_all_devices)
