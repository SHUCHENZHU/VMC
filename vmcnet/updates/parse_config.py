"""Get update functions from ConfigDicts."""
from typing import Tuple

from kfac_jax import Optimizer as kfac_Optimizer
import optax
from ml_collections import ConfigDict
import math
import vmcnet.mcmc.position_amplitude_core as pacore
import vmcnet.physics as physics
import vmcnet.utils as utils
import jax.lax as lax
import jax
import vmcnet.utils.curvature_tags_and_blocks as curvature_tags_and_blocks  # key of KFAC


from vmcnet.utils.distribute import (
broadcast_all_local_devices,
replicate_all_local_devices,
make_different_rng_key_on_all_devices,    
)


from vmcnet.utils.typing import (
    D,
    GetPositionFromData,
    LearningRateSchedule,
    ModelApply,
    OptimizerState,
    P,
    Array,
    PRNGKey,
    UpdateDataFn,
)

from .params import (
    UpdateParamFn,
    create_grad_energy_update_param_fn,
    create_kfac_update_param_fn,
    create_w2_kfac_update_param_fn,
)
from .sr import get_fisher_inverse_fn, constrain_norm, SRMode
from .spring import (
    get_spring_update_fn,
    constrain_norm as constrain_norm_spring,
)
from .spring_back import (
    get_spring_back_update_fn,
    constrain_norm as constrain_norm_spring_back,
)
from .racc import (
    get_racc_update_fn,
    constrain_norm as constrain_norm_racc,
)
from .W2 import (
    get_w2_kfac_loss,
    get_batch_raveled_gradient_local_energy,
    get_batch_gradient_local_energy,
    _get_dt_schedule,
    _get_lrV_schedule,
)
from .fisher_acc import (
    get_fisher_acc_update_fn,
    constrain_norm as constrain_norm_fisher_acc,
)
from .racc_cotangent import (
    get_racc_cotangent_update_fn,
    constrain_norm as constrain_norm_racc_cotangent,
)
import jax
import jax.numpy as jnp

from jax.tree_util import tree_map, tree_flatten, tree_unflatten
def _get_learning_rate_schedule(
    optimizer_config: ConfigDict,
) -> LearningRateSchedule:
    if optimizer_config.schedule_type == "constant":

        def learning_rate_schedule(t):
            return optimizer_config.learning_rate

    elif optimizer_config.schedule_type == "inverse_time":

        def learning_rate_schedule(t):
            return optimizer_config.learning_rate / (
                1.0 + optimizer_config.learning_decay_rate * t
            )
    elif optimizer_config.schedule_type == "square_inverse_time":

        def learning_rate_schedule(t):
            return optimizer_config.learning_rate / (
                1.0 + optimizer_config.learning_decay_rate * t
            )**2
    elif optimizer_config.schedule_type == "exponetial_decay":

        def learning_rate_schedule(t):
            power = jnp.array(t / optimizer_config.T, dtype=int)
            return optimizer_config.learning_rate *(
                optimizer_config.learning_decay_rate
            )**power
    elif optimizer_config.schedule_type == "sqrt_inverse_time":

        def learning_rate_schedule(t):
            return optimizer_config.learning_rate / jnp.sqrt((
                1.0 + optimizer_config.learning_decay_rate * t
            ))
    elif optimizer_config.schedule_type == "increase_time":

        def learning_rate_schedule(t):
            return optimizer_config.learning_rate * (
                1.0 + optimizer_config.learning_decay_rate * t
            )
    else:
        raise ValueError(
            "Learning rate schedule type not supported; {} was requested".format(
                optimizer_config.schedule_type
            )
        )

    return learning_rate_schedule

def _get_learning_rate_schedule1(
    optimizer_config: ConfigDict,
) -> LearningRateSchedule:
    if optimizer_config.schedule_type1 == "constant":

        def learning_rate_schedule(t):
            return optimizer_config.learning_rate1

    elif optimizer_config.schedule_type1 == "inverse_time":

        def learning_rate_schedule(t):
            return optimizer_config.learning_rate1 / (
                1.0 + optimizer_config.learning_decay_rate1 * t
            )
    elif optimizer_config.schedule_type1 == "square_inverse_time":

        def learning_rate_schedule(t):
            return optimizer_config.learning_rate1 / (
                1.0 + optimizer_config.learning_decay_rate1 * t
            )**2
    elif optimizer_config.schedule_type1 == "sqrt_inverse_time":

        def learning_rate_schedule(t):
            return optimizer_config.learning_rate1 / jnp.sqrt((
                1.0 + optimizer_config.learning_decay_rate1 * t
            ))
    elif optimizer_config.schedule_type1 == "exponetial_decay":

        def learning_rate_schedule(t):
            power = jnp.array(t / optimizer_config.T, dtype=int)
            return optimizer_config.learning_rate1 *(
                optimizer_config.learning_decay_rate1
            )**power
    elif optimizer_config.schedule_type1 == "increase_time":

        def learning_rate_schedule(t):
            a=optimizer_config.learning_rate1 * (
                1.0 + optimizer_config.learning_decay_rate1 * t
            )
            return jnp.minimum(a,optimizer_config.learning_rate1_upper)
       
    else:
        raise ValueError(
            "Learning rate schedule type not supported; {} was requested".format(
                optimizer_config.schedule_type1
            )
        )

    return learning_rate_schedule
def _get_learning_rate_schedule2(
    optimizer_config: ConfigDict,
) -> LearningRateSchedule:
    if optimizer_config.schedule_type2 == "constant":

        def learning_rate_schedule(t):
            return optimizer_config.learning_rate2

    elif optimizer_config.schedule_type2 == "inverse_time":

        def learning_rate_schedule(t):
            return optimizer_config.learning_rate2 / (
                1.0 + optimizer_config.learning_decay_rate2 * t
            )
    elif optimizer_config.schedule_type2 == "square_inverse_time":

        def learning_rate_schedule(t):
            return optimizer_config.learning_rate2 / (
                1.0 + optimizer_config.learning_decay_rate2 * t
            )**2
    elif optimizer_config.schedule_type2 == "sqrt_inverse_time":

        def learning_rate_schedule(t):
            return optimizer_config.learning_rate2 / jnp.sqrt((
                1.0 + optimizer_config.learning_decay_rate2 * t
            ))
    elif optimizer_config.schedule_type2 == "exponetial_decay":

        def learning_rate_schedule(t):
            power = jnp.array(t / optimizer_config.T, dtype=int)
            return optimizer_config.learning_rate2 *(
                optimizer_config.learning_decay_rate2
            )**power
    elif optimizer_config.schedule_type2 == "increase_time":

        def learning_rate_schedule(t):
            a=optimizer_config.learning_rate2 * (
                1.0 + optimizer_config.learning_decay_rate2 * t
            )
            return jnp.minimum(a,optimizer_config.learning_rate2_upper)
    else:
        raise ValueError(
            "Learning rate schedule type not supported; {} was requested".format(
                optimizer_config.schedule_type2
            )
        )

    return learning_rate_schedule
def _get_alpha_schedule(
    optimizer_config: ConfigDict,
) -> LearningRateSchedule:
    if optimizer_config.alpha_schedule_type == "constant":

        def alpha_schedule(t):
            return optimizer_config.alpha

    elif optimizer_config.alpha_schedule_type == "inverse_time":

        def alpha_schedule(t):
            return optimizer_config.alpha / (
                1.0 + optimizer_config.alpha_decay_rate * t
            )
    elif optimizer_config.alpha_schedule_type == "inverse_time_lower_bound":

        def alpha_schedule(t):
            return jnp.maximum(optimizer_config.alpha / (1.0 + optimizer_config.alpha_decay_rate * t),optimizer_config.alpha_lower_bound)
    elif optimizer_config.alpha_schedule_type == "sqrt_inverse_time":

        def alpha_schedule(t):
            return optimizer_config.alpha / jnp.sqrt((
                1.0 + optimizer_config.alpha_decay_rate * t
            ))
    elif optimizer_config.alpha_schedule_type == "increase_time":

        def alpha_schedule(t):
            a=optimizer_config.alpha * (
                1.0 + optimizer_config.alpha_decay_rate * t
            )
            return a
    elif optimizer_config.alpha_schedule_type == "exponetial_decay":

        def alpha_schedule(t):
            power = jnp.array(t / optimizer_config.T, dtype=int)
            return optimizer_config.alpha *(
                optimizer_config.alpha_decay_rate
            )**power
    else:
        raise ValueError(
            "alpha schedule type not supported; {} was requested".format(
                optimizer_config.alpha
            )
        )

    return alpha_schedule
def _get_beta_schedule(
    optimizer_config: ConfigDict,
) -> LearningRateSchedule:
    if optimizer_config.beta_schedule_type == "constant":

        def beta_schedule(t):
            return optimizer_config.beta

    elif optimizer_config.beta_schedule_type == "inverse_time":

        def beta_schedule(t):
            return optimizer_config.beta / (
                1.0 + optimizer_config.beta_decay_rate * t
            )
    elif optimizer_config.beta_schedule_type == "inverse_time_lower_bound":

        def beta_schedule(t):
            return jnp.maximum(optimizer_config.beta / (1.0 + optimizer_config.beta_decay_rate * t),optimizer_config.beta_lower_bound)
    elif optimizer_config.beta_schedule_type == "sqrt_inverse_time":

        def beta_schedule(t):
            return optimizer_config.beta / jnp.sqrt((
                1.0 + optimizer_config.beta_decay_rate * t
            ))
    elif optimizer_config.beta_schedule_type == "increase_time":

        def beta_schedule(t):
            a=optimizer_config.beta * (
                1.0 + optimizer_config.beta_decay_rate * t
            )
            return jnp.minimum(a,optimizer_config.beta_upper)
    elif optimizer_config.beta_schedule_type == "exponetial_decay":

        def beta_schedule(t):
            power = jnp.array(t / optimizer_config.T, dtype=int)
            return optimizer_config.beta *(
                optimizer_config.beta_decay_rate
            )**power
    else:
        raise ValueError(
            "beta schedule type not supported; {} was requested".format(
                optimizer_config.beta
            )
        )

    return beta_schedule


def get_update_fn_and_init_optimizer(
    log_psi_apply: ModelApply[P],
    vmc_config: ConfigDict,
    params: P,
    data: D,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    key: PRNGKey,
    apply_pmap: bool = True,
    energy_data_val=None,#TBD
) -> Tuple[UpdateParamFn[P, D, OptimizerState], OptimizerState, PRNGKey]:
    """Get an update function and initialize optimizer state from the vmc configuration.

    Args:
        log_psi_apply (Callable): computes log|psi(x)|, where the signature of this
            function is (params, x) -> log|psi(x)|
        vmc_config (ConfigDict): configuration for VMC
        params (pytree): params with which to initialize optimizer state
        data (pytree): data with which to initialize optimizer state
        get_position_fn (Callable): function which gets the position array from the data
        update_data_fn (Callable): function which updates data for new params
        energy_data_val_and_grad (Callable): function which computes the clipped energy
            value and gradient. Has the signature
                (params, x)
                -> ((expected_energy, auxiliary_energy_data), grad_energy),
            where auxiliary_energy_data is the tuple
            (expected_variance, local_energies, unclipped_energy, unclipped_variance)
        key (PRNGKey): PRNGKey with which to initialize optimizer state
        apply_pmap (bool, optional): whether to pmap the optimizer steps. Defaults to
            True.

    Raises:
        ValueError: A non-supported optimizer type is requested. Currently, KFAC, Adam,
            SGD, and SR (with either Adam or SGD) is supported.

    Returns:
        (UpdateParamFn, OptimizerState, PRNGKey):
        update param function with signature
            (params, data, optimizer_state, key)
            -> (new params, new state, metrics, new key),
        initial optimizer state, and
        PRNGKey
    """
    learning_rate_schedule = _get_learning_rate_schedule(
        vmc_config.optimizer[vmc_config.optimizer_type]
    )
    
    print("pmap",apply_pmap)
    
    
    if  vmc_config.optimizer_type == "fisher_acc":
        learning_rate_schedule1 = _get_learning_rate_schedule1(
            vmc_config.optimizer[vmc_config.optimizer_type]
        )
        learning_rate_schedule2 = _get_learning_rate_schedule2(
            vmc_config.optimizer[vmc_config.optimizer_type]
        )
        beta_schedule = _get_beta_schedule(
        vmc_config.optimizer[vmc_config.optimizer_type]
        )
    alpha_schedule = _get_alpha_schedule(
        vmc_config.optimizer[vmc_config.optimizer_type]
    )
    if vmc_config.optimizer_type == "kfac":
        return get_kfac_update_fn_and_state(
            params,
            data,
            get_position_fn,
            update_data_fn,
            energy_data_val_and_grad,
            key,
            learning_rate_schedule,
            vmc_config.optimizer.kfac,
            vmc_config.record_param_l1_norm,
            apply_pmap=apply_pmap,
        )
    elif vmc_config.optimizer_type == "sgd":
        (
            update_param_fn,
            optimizer_state,
        ) = get_sgd_update_fn_and_state(
            params,
            get_position_fn,
            update_data_fn,
            energy_data_val_and_grad,
            learning_rate_schedule,
            vmc_config.optimizer.sgd,
            vmc_config.record_param_l1_norm,
            apply_pmap=apply_pmap,
        )
        return update_param_fn, optimizer_state, key
    elif vmc_config.optimizer_type == "adam":
        (
            update_param_fn,
            optimizer_state,
        ) = get_adam_update_fn_and_state(
            params,
            get_position_fn,
            update_data_fn,
            energy_data_val_and_grad,
            learning_rate_schedule,
            vmc_config.optimizer.adam,
            vmc_config.record_param_l1_norm,
            apply_pmap=apply_pmap,
        )
        return update_param_fn, optimizer_state, key
    elif vmc_config.optimizer_type == "sr":
        (
            update_param_fn,
            optimizer_state,
        ) = get_sr_update_fn_and_state(
            log_psi_apply,
            params,
            get_position_fn,
            update_data_fn,
            energy_data_val_and_grad,
            learning_rate_schedule,
            vmc_config.optimizer.sr,
            vmc_config.optimizer[vmc_config.optimizer.sr.descent_type],
            vmc_config.record_param_l1_norm,
            apply_pmap=apply_pmap,
            nan_safe=vmc_config.nan_safe,
        )
        return update_param_fn, optimizer_state, key

    elif vmc_config.optimizer_type == "spring":
        (
            update_param_fn,
            optimizer_state,
        ) = get_spring_update_fn_and_state(
            log_psi_apply,
            params,
            get_position_fn,
            update_data_fn,
            energy_data_val_and_grad,
            learning_rate_schedule,
            vmc_config.optimizer.spring,
            vmc_config.record_param_l1_norm,
            apply_pmap=apply_pmap,
        )
        return update_param_fn, optimizer_state, key
    elif vmc_config.optimizer_type == "spring_back":
        (
            update_param_fn,
            optimizer_state,
        ) = get_spring_back_update_fn_and_state(
            log_psi_apply,
            params,
            get_position_fn,
            update_data_fn,
            energy_data_val_and_grad,
            learning_rate_schedule,
            vmc_config.optimizer.spring_back,
            vmc_config.record_param_l1_norm,
            apply_pmap=apply_pmap,
        )
        return update_param_fn, optimizer_state, key
    elif vmc_config.optimizer_type == "racc":
        (
            update_param_fn,
            optimizer_state,
        ) = get_racc_update_fn_and_state(
            log_psi_apply,
            params,
            get_position_fn,
            update_data_fn,
            energy_data_val_and_grad,
            learning_rate_schedule,
            alpha_schedule,
            vmc_config.optimizer.racc,
            vmc_config.record_param_l1_norm,
            apply_pmap=apply_pmap,
        )
        return update_param_fn, optimizer_state, key
    elif vmc_config.optimizer_type == "racc_cotangent":
        
        (
            update_param_fn,
            optimizer_state,
        ) = get_racc_cotangent_update_fn_and_state(
            log_psi_apply,
            params,
            get_position_fn,
            update_data_fn,
            energy_data_val_and_grad,
            learning_rate_schedule,
            alpha_schedule,
            vmc_config.optimizer.racc_cotangent,
            vmc_config.record_param_l1_norm,
            apply_pmap=apply_pmap,
        )
        return update_param_fn, optimizer_state, key
    elif vmc_config.optimizer_type == "w2":
        from vmcnet.train.runners import _get_clipping_fn
        learning_rate_schedule_V=_get_lrV_schedule(vmc_config.optimizer[vmc_config.optimizer_type])
        dt_schedule=_get_dt_schedule(vmc_config.optimizer[vmc_config.optimizer_type])
        alpha_schedule=_get_alpha_schedule(vmc_config.optimizer[vmc_config.optimizer_type])
        clipping_fn=_get_clipping_fn(vmc_config)
        return get_w2_kfac_update_fn_and_state(
            params,
            data,
            energy_data_val,
            energy_data_val_and_grad,
            get_position_fn,
            update_data_fn,
            log_psi_apply,
            key,
            learning_rate_schedule,
            learning_rate_schedule_V,
            dt_schedule,
            alpha_schedule,
            vmc_config.optimizer.w2,
            clipping_fn,
            vmc_config.record_param_l1_norm,
            apply_pmap=apply_pmap,
        )
    elif vmc_config.optimizer_type == "fisher_acc":
        (
            update_param_fn,
            optimizer_state,
        ) = get_fisher_acc_update_fn_and_state(
            log_psi_apply,
            params,
            get_position_fn,
            update_data_fn,
            energy_data_val_and_grad,
            energy_data_val,
            learning_rate_schedule1,
            learning_rate_schedule2,
            alpha_schedule,
            beta_schedule,
            vmc_config.optimizer.fisher_acc,
            vmc_config.record_param_l1_norm,
            apply_pmap=apply_pmap,
        )
        return update_param_fn, optimizer_state, key
        
    else:
        raise ValueError(
            "Requested optimizer type not supported; {} was requested".format(
                vmc_config.optimizer_type
            )
        )


def get_kfac_update_fn_and_state(
    params: P,
    data: D,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    key: PRNGKey,
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, OptimizerState], OptimizerState, PRNGKey]:
    """Get an update param function, initial state, and key for KFAC.

    Args:
        params (pytree): params with which to initialize optimizer state
        data (pytree): data with which to initialize optimizer state
        get_position_fn (Callable): function which gets the position array from the data
        update_data_fn (Callable): function which updates data for new params
        energy_data_val_and_grad (Callable): function which computes the clipped energy
            value and gradient. Has the signature
                (params, x)
                -> ((expected_energy, auxiliary_energy_data), grad_energy),
            where auxiliary_energy_data is the tuple
            (expected_variance, local_energies, unclipped_energy, unclipped_variance)
        key (PRNGKey): PRNGKey with which to initialize optimizer state
        learning_rate_schedule (Callable): function which returns a learning rate from
            epoch number. Has signature epoch -> learning_rate
        optimizer_config (ConfigDict): configuration for KFAC
        record_param_l1_norm (bool, optional): whether to record the L1 norm of the
            parameters in the metrics. Defaults to False.
        apply_pmap (bool, optional): whether to pmap the optimizer steps. Defaults to
            True.

    Returns:
        (UpdateParamFn, kfac_opt.State, PRNGKey):
        update param function with signature
            (params, data, optimizer_state, key)
            -> (new params, new state, metrics, new key),
        initial optimizer state, and
        PRNGKey
    """
    
    
    optimizer = kfac_Optimizer(
        energy_data_val_and_grad,
        l2_reg=optimizer_config.l2_reg,
        norm_constraint=optimizer_config.norm_constraint,
        value_func_has_aux=True,
        value_func_has_rng=True,
        learning_rate_schedule=learning_rate_schedule,
        curvature_ema=optimizer_config.curvature_ema,
        inverse_update_period=optimizer_config.inverse_update_period,
        min_damping=optimizer_config.min_damping,
        num_burnin_steps=0,
        register_only_generic=optimizer_config.register_only_generic,
        estimation_mode=optimizer_config.estimation_mode,
        multi_device=apply_pmap,
        pmap_axis_name=utils.distribute.PMAP_AXIS_NAME,
        # Mypy can't find GRAPH_PATTERNS because we've ignored types in the curvature
        # tags file since it's not typed properly.
        auto_register_kwargs=dict(
            graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERNS,  # type: ignore
        ),
    )
    key, subkey = utils.distribute.split_or_psplit_key(key, apply_pmap)

    optimizer_state = optimizer.init(params, subkey, get_position_fn(data))
    

    
    update_param_fn = create_kfac_update_param_fn(
        optimizer,
        optimizer_config.damping,
        pacore.get_position_from_data,
        update_data_fn,
        record_param_l1_norm=record_param_l1_norm,
    )

    return update_param_fn, optimizer_state, key


class kfac_optimizer_w2():
    def __init__(self, base_optimizer,apply_pmap):
        self.base_optimizer = base_optimizer
        self.iteration=0
        self.multi_device=apply_pmap  
    def initialize(self,params,positions, key): 
        
         
        
        velocity=0.0*positions
            

        batch_kfac=(positions, positions, velocity)
        #batch=broadcast_all_local_devices(batch)
        #base_state=self.base_optimizer.init(replicate_all_local_devices(params), key,  batch    )  #
        base_state=self.base_optimizer.init(params, key,  batch_kfac )  #
        return (base_state, velocity,positions)

    def update(self, params,state,rng,batch,momentum,damping): 
        
        base_state=state[0]
        
        
        params, new_base_state, stats = self.base_optimizer.step(  
            params=params,
            state=base_state,
            rng=rng,
            batch=batch,
            #data_iterator=iter([batch]),
            momentum=momentum,
            damping=damping,
            global_step_int=self.iteration, # optimizer_state.step_counter,
        )
        self.iteration+=1

        return new_base_state, params,stats

#optimizer_state=_init_optimizer_w2(optimizer, params,positions, subkey, apply_pmap)
def _init_optimizer_w2(
    optimizer:  kfac_optimizer_w2, params: P,positions,key
) :
    optimizer_init = optimizer.initialize
#     if apply_pmap:
#         optimizer_init = utils.distribute.pmap(optimizer_init)
    optimizer_state = optimizer_init(params,positions,key)
    return optimizer_state

def get_w2_kfac_update_fn_and_state(
    params: P,
    data: D,
    local_energy_fn,
    energy_data_val_and_grad,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    log_psi_apply,
    key: PRNGKey,
    learning_rate_schedule_params: LearningRateSchedule,
    learning_rate_schedule_V: LearningRateSchedule,
    dt_schedule: LearningRateSchedule,
    alpha_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    clipping_fn,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, OptimizerState], OptimizerState, PRNGKey]:
    """Get an update param function, initial state, and key for KFAC.

    Args:
        params (pytree): params with which to initialize optimizer state
        data (pytree): data with which to initialize optimizer state
        get_position_fn (Callable): function which gets the position array from the data
        update_data_fn (Callable): function which updates data for new params
        energy_data_val_and_grad (Callable): function which computes the clipped energy
            value and gradient. Has the signature
                (params, x)
                -> ((expected_energy, auxiliary_energy_data), grad_energy),
            where auxiliary_energy_data is the tuple
            (expected_variance, local_energies, unclipped_energy, unclipped_variance)
        key (PRNGKey): PRNGKey with which to initialize optimizer state
        learning_rate_schedule (Callable): function which returns a learning rate from
            epoch number. Has signature epoch -> learning_rate
        optimizer_config (ConfigDict): configuration for KFAC
        record_param_l1_norm (bool, optional): whether to record the L1 norm of the
            parameters in the metrics. Defaults to False.
        apply_pmap (bool, optional): whether to pmap the optimizer steps. Defaults to
            True.

    Returns:
        (UpdateParamFn, kfac_opt.State, PRNGKey):
        update param function with signature
            (params, data, optimizer_state, key)
            -> (new params, new state, metrics, new key),
        initial optimizer state, and
        PRNGKey
    """
    loss_w2 =get_w2_kfac_loss(log_psi_apply,local_energy_fn,clipping_fn)
  
    
    batch_gradient_local_energy=get_batch_gradient_local_energy(local_energy_fn)

    base_optimizer = kfac_Optimizer(
        jax.value_and_grad(loss_w2, argnums=0,has_aux=True),
        #energy_data_val_and_grad,
        l2_reg=0.0,
        norm_constraint=optimizer_config.norm_constraint,
        value_func_has_aux=True,  
        value_func_has_rng=True,  
        learning_rate_schedule=learning_rate_schedule_params,
        curvature_ema=optimizer_config.curvature_ema,
        inverse_update_period=optimizer_config.inverse_update_period,
        min_damping=optimizer_config.min_damping,
        num_burnin_steps=0,
        register_only_generic=optimizer_config.register_only_generic,
        estimation_mode=optimizer_config.estimation_mode,
        multi_device=apply_pmap,
        pmap_axis_name=utils.distribute.PMAP_AXIS_NAME,
        # Mypy can't find GRAPH_PATTERNS because we've ignored types in the curvature
        # tags file since it's not typed properly.
        auto_register_kwargs=dict(
            graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERNS,  # type: ignore
        ),
    )
    key, subkey = utils.distribute.split_or_psplit_key(key, apply_pmap) 
    positions=get_position_fn(data)
    
    
    
    optimizer=kfac_optimizer_w2(base_optimizer,apply_pmap)
   
    optimizer_state=_init_optimizer_w2(optimizer, params,positions, subkey) 

    
    update_param_fn = create_w2_kfac_update_param_fn(
        optimizer,
        optimizer_config.damping,
        pacore.get_position_from_data,
        batch_gradient_local_energy,
        update_data_fn,
        learning_rate_schedule_V,
        dt_schedule,
        alpha_schedule,
        record_param_l1_norm=record_param_l1_norm,
        acceleration=optimizer_config.acceleration
         )

    return update_param_fn, optimizer_state, key





def _get_adam_optax_optimizer(
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
) -> optax.GradientTransformation:
    return optax.adam(
        learning_rate=learning_rate_schedule,
        b1=optimizer_config.b1,
        b2=optimizer_config.b2,
        eps=optimizer_config.eps,
        eps_root=optimizer_config.eps_root,
    )


def _get_sgd_optax_optimizer(
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
) -> optax.GradientTransformation:
    return optax.sgd(
        learning_rate=learning_rate_schedule,
        momentum=optimizer_config.momentum if optimizer_config.momentum != 0 else None,
        nesterov=optimizer_config.nesterov,
    )


def _init_optax_optimizer(
    optimizer: optax.GradientTransformation, params: P, apply_pmap: bool = True
) -> optax.OptState:
    optimizer_init = optimizer.init
    if apply_pmap:
        optimizer_init = utils.distribute.pmap(optimizer_init)
    optimizer_state = optimizer_init(params)
    return optimizer_state


def _get_optax_update_fn_and_state(
    optimizer: optax.GradientTransformation,
    params: P,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, optax.OptState], optax.OptState]:
    def optimizer_apply(grad, params, optimizer_state, data, aux):
        del data, aux
        updates, optimizer_state = optimizer.update(grad, optimizer_state, params)
        params = optax.apply_updates(params, updates)
        return params, optimizer_state

    update_param_fn = create_grad_energy_update_param_fn(
        energy_data_val_and_grad,
        optimizer_apply,
        get_position_fn=get_position_fn,
        update_data_fn=update_data_fn,
        record_param_l1_norm=record_param_l1_norm,
        apply_pmap=apply_pmap,
    )

    optimizer_state = _init_optax_optimizer(optimizer, params, apply_pmap=apply_pmap)

    return update_param_fn, optimizer_state


def get_adam_update_fn_and_state(
    params: P,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, optax.OptState], optax.OptState]:
    """Get an update param function and initial state for Adam.

    Args:
        params (pytree): params with which to initialize optimizer state
        get_position_fn (Callable): function which gets the position array from the data
        update_data_fn (Callable): function which updates data for new params
        energy_data_val_and_grad (Callable): function which computes the clipped energy
            value and gradient. Has the signature
                (params, x)
                -> ((expected_energy, auxiliary_energy_data), grad_energy),
            where auxiliary_energy_data is the tuple
            (expected_variance, local_energies, unclipped_energy, unclipped_variance)
        learning_rate_schedule (Callable): function which returns a learning rate from
            epoch number. Has signature epoch -> learning_rate
        optimizer_config (ConfigDict): configuration for Adam
        record_param_l1_norm (bool, optional): whether to record the L1 norm of the
            parameters in the metrics. Defaults to False.
        apply_pmap (bool, optional): whether to pmap the optimizer steps. Defaults to
            True.

    Returns:
        (UpdateParamFn, optax.OptState):
        update param function with signature
            (params, data, optimizer_state, key)
            -> (new params, new state, metrics, new key), and
        initial optimizer state
    """
    optimizer = _get_adam_optax_optimizer(learning_rate_schedule, optimizer_config)

    return _get_optax_update_fn_and_state(
        optimizer,
        params,
        get_position_fn,
        update_data_fn,
        energy_data_val_and_grad,
        record_param_l1_norm,
        apply_pmap,
    )


def get_sgd_update_fn_and_state(
    params: P,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, optax.OptState], optax.OptState]:
    """Get an update param function and initial state for SGD.

    Args:
        params (pytree): params with which to initialize optimizer state
        get_position_fn (Callable): function which gets the position array from the data
        update_data_fn (Callable): function which updates data for new params
        energy_data_val_and_grad (Callable): function which computes the clipped energy
            value and gradient. Has the signature
                (params, x)
                -> ((expected_energy, auxiliary_energy_data), grad_energy),
            where auxiliary_energy_data is the tuple
            (expected_variance, local_energies, unclipped_energy, unclipped_variance)
        learning_rate_schedule (Callable): function which returns a learning rate from
            epoch number. Has signature epoch -> learning_rate
        optimizer_config (ConfigDict): configuration for SGD
        record_param_l1_norm (bool, optional): whether to record the L1 norm of the
            parameters in the metrics. Defaults to False.
        apply_pmap (bool, optional): whether to pmap the optimizer steps. Defaults to
            True.

    Returns:
        (UpdateParamFn, optax.OptState):
        update param function with signature
            (params, data, optimizer_state, key)
            -> (new params, new state, metrics, new key), and
        initial optimizer state
    """
    optimizer = _get_sgd_optax_optimizer(learning_rate_schedule, optimizer_config)

    return _get_optax_update_fn_and_state(
        optimizer,
        params,
        get_position_fn,
        update_data_fn,
        energy_data_val_and_grad,
        record_param_l1_norm,
        apply_pmap,
    )



def get_sr_update_fn_and_state(
    log_psi_apply: ModelApply[P],
    params: P,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    descent_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
    nan_safe: bool = True,
) -> Tuple[UpdateParamFn[P, D, optax.OptState], optax.OptState]:
    """
    Get an update param function and initial state for stochastic reconfiguration.

    Args:
        log_psi_apply (Callable): computes log|psi(x)|, where the signature of this
            function is (params, x) -> log|psi(x)|
        params (pytree): params with which to initialize optimizer state
        get_position_fn (Callable): function which gets the position array from the data
        update_data_fn (Callable): function which updates data for new params
        energy_data_val_and_grad (Callable): function which computes the clipped energy
            value and gradient. Has the signature
                (params, x)
                -> ((expected_energy, auxiliary_energy_data), grad_energy),
            where auxiliary_energy_data is the tuple
            (expected_variance, local_energies, unclipped_energy, unclipped_variance)
        learning_rate_schedule (Callable): function which returns a learning rate from
            epoch number. Has signature epoch -> learning_rate
        optimizer_config (ConfigDict): configuration for stochastic reconfiguration
        descent_config (ConfigDict): configuration for the gradient descent-like method
            used to apply the preconditioned updates
        record_param_l1_norm (bool, optional): whether to record the L1 norm of the
            parameters in the metrics. Defaults to False.
        apply_pmap (bool, optional): whether to pmap the optimizer steps. Defaults to
            True.
        nan_safe (bool, optional): whether the mean function used when centering the
            Jacobian of log|psi(x)| during the Fisher matvec is nan-safe. Defaults to
            True.

    Raises:
        ValueError: A non-supported descent type is requested. Currently only Adam and
            SGD are supported.

    Returns:
        (UpdateParamFn, optax.OptState):
        update param function with signature
            (params, data, optimizer_state, key)
            -> (new params, new state, metrics, new key), and
        initial optimizer state
    """
    maxiter = optimizer_config.maxiter if optimizer_config.maxiter >= 0 else None
    mean_grad_fn = utils.distribute.get_mean_over_first_axis_fn(nan_safe=nan_safe)
    precondition_grad_fn = get_fisher_inverse_fn(
        log_psi_apply,
        mean_grad_fn,
        damping=optimizer_config.damping,
        maxiter=maxiter,
        mode=SRMode[optimizer_config.mode.upper()],
    )

    if optimizer_config.descent_type == "adam":
        descent_optimizer = _get_adam_optax_optimizer(
            learning_rate_schedule, descent_config
        )
    elif optimizer_config.descent_type == "sgd":
        descent_optimizer = _get_sgd_optax_optimizer(
            learning_rate_schedule, descent_config
        )
    else:
        raise ValueError(
            "Requested descent type not supported; {} was requested".format(
                optimizer_config.descent_type
            )
        )

    def get_optimizer_step_count(optimizer_state):
        return optimizer_state[1].count

    def optimizer_apply(grad, params, optimizer_state, data, aux):
        preconditioned_grad = precondition_grad_fn(grad, params, get_position_fn(data))
        step_count = get_optimizer_step_count(optimizer_state)
        learning_rate = learning_rate_schedule(step_count)
        constrained_grad = constrain_norm(
            grad, preconditioned_grad, learning_rate, optimizer_config.norm_constraint
        )

        updates, optimizer_state = descent_optimizer.update(
            constrained_grad, optimizer_state, params
        )
        params = optax.apply_updates(params, updates)
        return params, optimizer_state

    update_param_fn = create_grad_energy_update_param_fn(
        energy_data_val_and_grad,
        optimizer_apply,
        get_position_fn=get_position_fn,
        update_data_fn=update_data_fn,
        record_param_l1_norm=record_param_l1_norm,
        apply_pmap=apply_pmap,
    )
    optimizer_state = _init_optax_optimizer(
        descent_optimizer, params, apply_pmap=apply_pmap
    )

    
    return update_param_fn, optimizer_state



def get_spring_update_fn_and_state(
    log_psi_apply: ModelApply[P],
    params: P,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, optax.OptState], optax.OptState]:
    
    """
    Get an update param function and initial state for SPRING.

    Args:
        log_psi_apply (Callable): computes log|psi(x)|, where the signature of this
            function is (params, x) -> log|psi(x)|
        params (pytree): params with which to initialize optimizer state
        get_position_fn (Callable): function which gets the position array from the data
        update_data_fn (Callable): function which updates data for new params
        energy_data_val_and_grad (Callable): function which computes the clipped energy
            value and gradient. Has the signature
                (params, x)
                -> ((expected_energy, auxiliary_energy_data), grad_energy),
            where auxiliary_energy_data is the tuple
            (expected_variance, local_energies, unclipped_energy, unclipped_variance)
        learning_rate_schedule (Callable): function which returns a learning rate from
            epoch number. Has signature epoch -> learning_rate
        optimizer_config (ConfigDict): configuration for stochastic reconfiguration
        record_param_l1_norm (bool, optional): whether to record the L1 norm of the
            parameters in the metrics. Defaults to False.
        apply_pmap (bool, optional): whether to pmap the optimizer steps. Defaults to
            True.

    Returns:
        (UpdateParamFn, optax.OptState):
        update param function with signature
            (params, data, optimizer_state, key)
            -> (new params, new state, metrics, new key), and
        initial optimizer state
    """
    spring_update_fn = get_spring_update_fn(
        log_psi_apply,
        optimizer_config.damping,
        optimizer_config.mu,
        optimizer_config.momentum,
    )

    descent_optimizer = optax.sgd(
        learning_rate=learning_rate_schedule, momentum=0, nesterov=False
    )

    def prev_update(optimizer_state):
        return optimizer_state[0].trace

    def optimizer_apply(regular_grad, params, optimizer_state, data, aux):
        del regular_grad
        grad = spring_update_fn(
            aux["centered_local_energies"],
            params,
            prev_update(optimizer_state),
            get_position_fn(data),
        )

        updates, optimizer_state = descent_optimizer.update(
            grad, optimizer_state, params
        )

        if optimizer_config.constrain_norm:
            updates = constrain_norm_spring(
                updates,
                optimizer_config.norm_constraint,
            )

        params = optax.apply_updates(params, updates)
        return params, optimizer_state

    update_param_fn = create_grad_energy_update_param_fn(
        energy_data_val_and_grad,
        optimizer_apply,
        get_position_fn=get_position_fn,
        update_data_fn=update_data_fn,
        record_param_l1_norm=record_param_l1_norm,
        apply_pmap=apply_pmap,
    )
    optimizer_state = _init_optax_optimizer(
        descent_optimizer, params, apply_pmap=apply_pmap
    )

    return update_param_fn, optimizer_state

def get_spring_back_update_fn_and_state(
    log_psi_apply: ModelApply[P],
    params: P,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, optax.OptState], optax.OptState]:
    """Get an update param function and initial state for SPRING.

    Args:
        log_psi_apply (Callable): computes log|psi(x)|, where the signature of this
            function is (params, x) -> log|psi(x)|
        params (pytree): params with which to initialize optimizer state
        get_position_fn (Callable): function which gets the position array from the data
        update_data_fn (Callable): function which updates data for new params
        energy_data_val_and_grad (Callable): function which computes the clipped energy
            value and gradient. Has the signature
                (params, x)
                -> ((expected_energy, auxiliary_energy_data), grad_energy),
            where auxiliary_energy_data is the tuple
            (expected_variance, local_energies, unclipped_energy, unclipped_variance)
        learning_rate_schedule (Callable): function which returns a learning rate from
            epoch number. Has signature epoch -> learning_rate
        optimizer_config (ConfigDict): configuration for stochastic reconfiguration
        record_param_l1_norm (bool, optional): whether to record the L1 norm of the
            parameters in the metrics. Defaults to False.
        apply_pmap (bool, optional): whether to pmap the optimizer steps. Defaults to
            True.

    Returns:
        (UpdateParamFn, optax.OptState):
        update param function with signature
            (params, data, optimizer_state, key)
            -> (new params, new state, metrics, new key), and
        initial optimizer state
    """
    spring_back_update_fn = get_spring_back_update_fn(
        log_psi_apply,
        optimizer_config.damping,
        optimizer_config.mu,
        optimizer_config.theta,
    )

    descent_optimizer = optax.sgd(
        learning_rate=learning_rate_schedule, momentum=0, nesterov=False
    )

    def prev_update(optimizer_state):
        return optimizer_state[0].trace

    def optimizer_apply(regular_grad, params, optimizer_state, data, aux):
        del regular_grad
        grad = spring_back_update_fn(
            aux["centered_local_energies"],
            params,
            prev_update(optimizer_state),
            get_position_fn(data),
        )

        updates, optimizer_state = descent_optimizer.update(
            grad, optimizer_state, params
        )

        if optimizer_config.constrain_norm:
            updates = constrain_norm_spring(
                updates,
                optimizer_config.norm_constraint,
            )

        params = optax.apply_updates(params, updates)
        return params, optimizer_state

    update_param_fn = create_grad_energy_update_param_fn(
        energy_data_val_and_grad,
        optimizer_apply,
        get_position_fn=get_position_fn,
        update_data_fn=update_data_fn,
        record_param_l1_norm=record_param_l1_norm,
        apply_pmap=apply_pmap,
    )
    optimizer_state = _init_optax_optimizer(
        descent_optimizer, params, apply_pmap=apply_pmap
    )

    return update_param_fn, optimizer_state



class CustomOptimizer():
    def __init__(self, base_optimizer):
        self.base_optimizer = base_optimizer

    def initialize(self, params):
        
        base_state = self.base_optimizer.init(params)
         
        #extra_state = tree_map(lambda p: jnp.zeros_like(p), params)
        #flat_params, _ = tree_flatten(params)
        flat_params, un_fn=jax.flatten_util.ravel_pytree(params)
        extra_state = un_fn(jnp.zeros(len(flat_params)))
        #extra_state = jnp.zeros(len(flat_params))
        
        return (base_state, extra_state)

    def new_update(self, updates, state, params, new_extra_state): 
        base_state=state[0]
        extra_state=state[-1]
        _, new_base_state = self.base_optimizer.update(updates, base_state, params)
        
       
        
        return (new_base_state,new_extra_state)

def _init_custom_optimizer(
    optimizer:  CustomOptimizer, params: P, apply_pmap: bool = True
) :
    optimizer_init = optimizer.initialize
    if apply_pmap:
        optimizer_init = utils.distribute.pmap(optimizer_init)
    optimizer_state = optimizer_init(params)
    return optimizer_state


def get_racc_update_fn_and_state(
    log_psi_apply: ModelApply[P],
    params: P,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    learning_rate_schedule: LearningRateSchedule,
    alpha_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
):
    # -> Tuple[UpdateParamFn[P, D, optax.OptState], optax.OptState]
    """Get an update param function and initial state for racc.

    Args:
        log_psi_apply (Callable): computes log|psi(x)|, where the signature of this
            function is (params, x) -> log|psi(x)|
        params (pytree): params with which to initialize optimizer state
        get_position_fn (Callable): function which gets the position array from the data
        update_data_fn (Callable): function which updates data for new params
        energy_data_val_and_grad (Callable): function which computes the clipped energy
            value and gradient. Has the signature
                (params, x)
                -> ((expected_energy, auxiliary_energy_data), grad_energy),
            where auxiliary_energy_data is the tuple
            (expected_variance, local_energies, unclipped_energy, unclipped_variance)
        learning_rate_schedule (Callable): function which returns a learning rate from
            epoch number. Has signature epoch -> learning_rate
        optimizer_config (ConfigDict): configuration for stochastic reconfiguration
        record_param_l1_norm (bool, optional): whether to record the L1 norm of the
            parameters in the metrics. Defaults to False.
        apply_pmap (bool, optional): whether to pmap the optimizer steps. Defaults to
            True.

    Returns:
        (UpdateParamFn, optax.OptState):
        update param function with signature
            (params, data, optimizer_state, key)
            -> (new params, new state, metrics, new key), and
        initial optimizer state
    """
    racc_update_fn = get_racc_update_fn(
        log_psi_apply,
        optimizer_config.damping)

    descent_optimizer0 = optax.sgd(
        learning_rate=learning_rate_schedule, momentum=0.0, nesterov=False
    )
    descent_optimizer=CustomOptimizer(descent_optimizer0)

    def optimizer_apply(regular_grad, params, optimizer_state, data, aux):
        del regular_grad
        epoch_step=optimizer_state[0][1].count
        sqrt_s=learning_rate_schedule(epoch_step)
        alpha=alpha_schedule(epoch_step)
        
#         pre1, unravel_fn = jax.flatten_util.ravel_pytree(optimizer_state[0][0].trace)
        
        eta = racc_update_fn(
            aux["centered_local_energies"],
            params,
            optimizer_state[-1], # prev_eta 
            get_position_fn(data),
            sqrt_s,
            alpha,
            optimizer_config.prev_eta_proj,
            optimizer_config.correction_rate,
            restart=optimizer_config.restart,
            key=jax.random.PRNGKey(epoch_step)

        )
       

        optimizer_state = descent_optimizer.new_update(
             eta,optimizer_state, params,eta
        )
       
        if optimizer_config.constrain_norm:
            eta_clip = constrain_norm_racc(
                eta,
                optimizer_config.norm_constraint,
            )
        else:
            eta_clip=eta
        updates=tree_map(lambda x: -sqrt_s * x, eta_clip)
        params = optax.apply_updates(params, updates)
     
        return params, optimizer_state

    update_param_fn = create_grad_energy_update_param_fn(
        energy_data_val_and_grad,
        optimizer_apply,
        get_position_fn=get_position_fn,
        update_data_fn=update_data_fn,
        record_param_l1_norm=record_param_l1_norm,
        apply_pmap=apply_pmap,
    )
    optimizer_state = _init_custom_optimizer(
        descent_optimizer, params, apply_pmap=apply_pmap
    )
   

    return update_param_fn, optimizer_state






class CustomOptimizer2():
    def __init__(self, base_optimizer):
        self.base_optimizer = base_optimizer

    def initialize(self, params):
        
        base_state = self.base_optimizer.init(params)
        
        #extra_state = tree_map(lambda p: jnp.zeros_like(p), params)
        #flat_params, _ = tree_flatten(params)
        flat_params, _=jax.flatten_util.ravel_pytree(params)
        extra_state = jnp.zeros(len(flat_params))
       
        
        return (base_state, extra_state)

    def new_update(self, updates, state, params, new_extra_state): 
        base_state=state[0]
        extra_state=state[-1]
        _, new_base_state = self.base_optimizer.update(updates, base_state, params)
        
       
        
        return (new_base_state,new_extra_state)

#         descent_optimizer0 = optax.sgd(
#         learning_rate=learning_rate_schedule, momentum=0.0, nesterov=False
#     )
#         descent_optimizer=CustomOptimizer(descent_optimizer0)
def _init_custom_optimizer2(
    optimizer:  CustomOptimizer, params: P, apply_pmap: bool = True
) :
    optimizer_init = optimizer.initialize
    if apply_pmap:
        optimizer_init = utils.distribute.pmap(optimizer_init)
    optimizer_state = optimizer_init(params)
    return optimizer_state


def get_racc_cotangent_update_fn_and_state(
    log_psi_apply: ModelApply[P],
    params: P,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    learning_rate_schedule: LearningRateSchedule,
    alpha_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
):
    # -> Tuple[UpdateParamFn[P, D, optax.OptState], optax.OptState]
    """Get an update param function and initial state for racc.

    Args:
        log_psi_apply (Callable): computes log|psi(x)|, where the signature of this
            function is (params, x) -> log|psi(x)|
        params (pytree): params with which to initialize optimizer state
        get_position_fn (Callable): function which gets the position array from the data
        update_data_fn (Callable): function which updates data for new params
        energy_data_val_and_grad (Callable): function which computes the clipped energy
            value and gradient. Has the signature
                (params, x)
                -> ((expected_energy, auxiliary_energy_data), grad_energy),
            where auxiliary_energy_data is the tuple
            (expected_variance, local_energies, unclipped_energy, unclipped_variance)
        learning_rate_schedule (Callable): function which returns a learning rate from
            epoch number. Has signature epoch -> learning_rate
        optimizer_config (ConfigDict): configuration for stochastic reconfiguration
        record_param_l1_norm (bool, optional): whether to record the L1 norm of the
            parameters in the metrics. Defaults to False.
        apply_pmap (bool, optional): whether to pmap the optimizer steps. Defaults to
            True.

    Returns:
        (UpdateParamFn, optax.OptState):
        update param function with signature
            (params, data, optimizer_state, key)
            -> (new params, new state, metrics, new key), and
        initial optimizer state
     """
    racc_cotangent_update_fn= get_racc_cotangent_update_fn(
        log_psi_apply,
        optimizer_config.damping)

    descent_optimizer0 = optax.sgd(
        learning_rate=learning_rate_schedule, momentum=0.0, nesterov=False
    )
    descent_optimizer=CustomOptimizer2(descent_optimizer0)

    def optimizer_apply(regular_grad, params, optimizer_state, data, aux):
        del regular_grad
        epoch_step=optimizer_state[0][1].count
        sqrt_s=learning_rate_schedule(epoch_step)
        alpha=alpha_schedule(epoch_step)
        
#         pre1, unravel_fn = jax.flatten_util.ravel_pytree(optimizer_state[0][0].trace)
        
        phi,eta = racc_cotangent_update_fn(
            aux["centered_local_energies"],
            params,
            optimizer_state[0][0].trace, # prev_eta 
            optimizer_state[-1], # prev_phi
            get_position_fn(data),
            sqrt_s,
            alpha,
            optimizer_config.correction_rate
        )
       
  #               new_update(self, updates, state, params, new_extra_state)
        if optimizer_config.constrain_norm:
            eta,phi = constrain_norm_racc_cotangent(
                eta,phi,
                optimizer_config.norm_constraint,
            )
        else:
            eta,phi=eta,phi
            
        optimizer_state = descent_optimizer.new_update(
             eta,optimizer_state, params,phi
        )
       
        
        updates=tree_map(lambda x: -sqrt_s * x, eta)
        params = optax.apply_updates(params, updates)
     
        return params, optimizer_state

    update_param_fn = create_grad_energy_update_param_fn(
        energy_data_val_and_grad,
        optimizer_apply,
        get_position_fn=get_position_fn,
        update_data_fn=update_data_fn,
        record_param_l1_norm=record_param_l1_norm,
        apply_pmap=apply_pmap,
    )
    optimizer_state = _init_custom_optimizer2(
        descent_optimizer, params, apply_pmap=apply_pmap
    )
   

    return update_param_fn, optimizer_state





class CustomOptimizer_pos():
    def __init__(self, base_optimizer,n_chains):
        self.base_optimizer = base_optimizer
        #self.cot_init=jnp.array(n_chains)
        self.cot_init=jnp.zeros(n_chains)

    def initialize(self, params): 
         
        base_state = self.base_optimizer.init(params)
         
        

        flat_params, un_fn=jax.flatten_util.ravel_pytree(params)
        prev_eta = un_fn(jnp.zeros(len(flat_params)))
        #prev_min_eta=jnp.zeros(len(flat_params))
        prev_cot= self.cot_init
        positions=jnp.expand_dims(jnp.expand_dims(self.cot_init, axis=-1), axis=-1) #initial (n_chains,1,1) tensor
        return (base_state, prev_eta,prev_cot,positions)

    def new_update(self, updates, state, params, new_eta,new_cot,new_positions): 
        base_state=state[0]
        
        _, new_base_state = self.base_optimizer.update(updates, base_state, params)
        
 
        return (new_base_state, new_eta,new_cot,new_positions)
    

def _init_custom_pos_optimizer(
    optimizer:  CustomOptimizer_pos, params: P, nchains_train: Array, apply_pmap: bool = True
) :
    optimizer_init = optimizer.initialize
   
    if apply_pmap:
        optimizer_init = utils.distribute.pmap(optimizer_init)
    optimizer_state = optimizer_init(params)
    return optimizer_state


def get_fisher_acc_update_fn_and_state(
    log_psi_apply: ModelApply[P],
    params: P,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    local_energy_data_val,
    learning_rate_schedule1: LearningRateSchedule,
    learning_rate_schedule2: LearningRateSchedule,
    alpha_schedule: LearningRateSchedule,
    beta_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
):
     #-> Tuple[UpdateParamFn[P, D, optax.OptState], optax.OptState]
    """Get an update param function and initial state for racc.

    Args:
        log_psi_apply (Callable): computes log|psi(x)|, where the signature of this
            function is (params, x) -> log|psi(x)|
        params (pytree): params with which to initialize optimizer state
        get_position_fn (Callable): function which gets the position array from the data
        update_data_fn (Callable): function which updates data for new params
        energy_data_val_and_grad (Callable): function which computes the clipped energy
            value and gradient. Has the signature
                (params, x)
                -> ((expected_energy, auxiliary_energy_data), grad_energy),
            where auxiliary_energy_data is the tuple
            (expected_variance, local_energies, unclipped_energy, unclipped_variance)
        learning_rate_schedule (Callable): function which returns a learning rate from
            epoch number. Has signature epoch -> learning_rate
        optimizer_config (ConfigDict): configuration for stochastic reconfiguration
        record_param_l1_norm (bool, optional): whether to record the L1 norm of the
            parameters in the metrics. Defaults to False.
        apply_pmap (bool, optional): whether to pmap the optimizer steps. Defaults to
            True.

    Returns:
        (UpdateParamFn, optax.OptState):
        update param function with signature
            (params, data, optimizer_state, key)
            -> (new params, new state, metrics, new key), and
        initial optimizer state
    """
    
    fisher_acc_update_fn = get_fisher_acc_update_fn(
        log_psi_apply,
        local_energy_data_val,
        optimizer_config.damping)

    
    epoch_warm=optimizer_config.epoch_warm
    epoch_T=optimizer_config.epoch_T
    nchains_train=optimizer_config.nchains_train
    
    descent_optimizer0 = optax.sgd(
        learning_rate=learning_rate_schedule1, momentum=0.0, nesterov=False
    )
    descent_optimizer=CustomOptimizer_pos(descent_optimizer0, nchains_train)
    
    
    
   

    def optimizer_apply(regular_grad, params, optimizer_state, data, aux):
        del regular_grad
        epoch_step=optimizer_state[0][1].count
        sqrt_s1=learning_rate_schedule1(epoch_step)
        prev_s1=learning_rate_schedule1(epoch_step-1)
        sqrt_s2=learning_rate_schedule2(epoch_step)
        alpha=alpha_schedule(epoch_step)
        beta=beta_schedule(epoch_step)
        mcmc_positions=get_position_fn(data)
        balance=optimizer_config.balance 
        mu=optimizer_config.mu  
        
      
        # positions = lax.cond(
        #     (epoch_step % epoch_T != 0) & (epoch_step > epoch_warm+1),
        #     lambda _: optimizer_state[-1]+jnp.zeros_like(mcmc_positions),  #  prev_positions,just for brevity
        #     lambda _: mcmc_positions,   
        #     operand=None   
        #     )

        positions = mcmc_positions

        
        
        eta,cot  = fisher_acc_update_fn( 
            epoch_step,
            aux["centered_local_energies"],
            params,
            optimizer_state[-3], # prev_eta
            optimizer_state[-2], # prev_cot 
            positions,          
            sqrt_s1,
            prev_s1,
            sqrt_s2,
            alpha,
            beta,
            balance,
            mu,
            optimizer_config.prev_eta_proj,
            restart=optimizer_config.restart,
            key=jax.random.PRNGKey(epoch_step),
            #mode=mode
            )#mixed
        
        
        if optimizer_config.constrain_norm:
            eta,cot = constrain_norm_fisher_acc(
                grad=eta,
                cot_vec=cot,
                cot_vec_threshold=optimizer_config.cot_element_constraint,
                norm_constraint=optimizer_config.norm_constraint,
            )
        
            
        updates=tree_map(lambda x: -sqrt_s1 * x, eta)
        params = optax.apply_updates(params, updates)    
            
        optimizer_state = descent_optimizer.new_update(
             updates,optimizer_state, params,eta,cot,positions
        )
       
        
        
     
        return params, optimizer_state
    
    
    
        

    update_param_fn = create_grad_energy_update_param_fn(
        energy_data_val_and_grad,
        optimizer_apply,
        get_position_fn=get_position_fn,  
        update_data_fn=update_data_fn,
        record_param_l1_norm=record_param_l1_norm,
        apply_pmap=apply_pmap,
    )
    
    optimizer_state = _init_custom_pos_optimizer(
        descent_optimizer, params,nchains_train, apply_pmap=apply_pmap
    )#None positions
   

    return update_param_fn, optimizer_state