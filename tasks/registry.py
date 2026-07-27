from __future__ import annotations

from typing import Dict, Type

from tasks.base import BehaviorTask
from tasks.cot_distortion import CoTDistortionTask
from tasks.honesty_control import HonestyControlTask
from tasks.motivated_reasoning import MotivatedReasoningTask
from tasks.reference_traffic import ReferenceTrafficTask
from tasks.sycophancy import SycophancyTask


TASK_REGISTRY: Dict[str, Type[BehaviorTask]] = {
    "reference_traffic": ReferenceTrafficTask,
    "sycophancy": SycophancyTask,
    "motivated_reasoning": MotivatedReasoningTask,
    "cot_distortion": CoTDistortionTask,
    "honesty_control": HonestyControlTask,
}
