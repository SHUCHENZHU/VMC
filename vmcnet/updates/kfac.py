"""Stochastic reconfiguration (SR) routine."""
from enum import Enum, auto
from typing import Callable, Optional

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy as jscp
import chex

from vmcnet.utils.distribute import pmean_if_pmap
from vmcnet.utils.typing import Array, ArrayLike, ModelApply, P

from vmcnet.utils.pytree_helpers import (
    multiply_tree_by_scalar,
    tree_inner_product,
    tree_sum,
) 
from vmcnet import utils


from kfac_jax import CurvatureEstimator


