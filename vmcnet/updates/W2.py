"""RACC routine."""
import jax
import jax.flatten_util
import jax.numpy as jnp
from jax import jvp, grad
from vmcnet.utils.typing import Array, ModelApply, P, Tuple
import ml_collections
import chex
from vmcnet.utils.pytree_helpers import (
    multiply_tree_by_scalar,
    tree_inner_product,
)
from vmcnet import utils
from jax import tree_util,jit,lax
#from vmcnet.physics.kinetic import create_laplacian_x_kinetic_energy
import kfac_jax

from ml_collections import ConfigDict
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


from vmcnet.utils.distribute import PMAP_AXIS_NAME






def clip_median(data):
    median = jnp.median(lax.all_gather(data, 'batch'))
    deviation = jnp.mean(jnp.abs(data - median))
    return jnp.clip(data, median - 5*deviation, median + 5*deviation)

def get_W2_update_fn(
    log_psi_apply: ModelApply[P],
    energy_data_val,
):
    """
    Get the racc update function.

    Args:
        log_psi_apply (Callable): computes log|psi(x)|, where the signature of this
            function is (params, x) -> log|psi(x)|
        damping (float): damping parameter
        mu (float): racc-specific regularization

    Returns:
        Callable: W2 update function. Has the signature
        (centered_energies, params, prev_grad, positions) -> new_grad
    """

    def raveled_log_psi_grad(params: P, positions: Array) -> Array:
        log_grads = jax.grad(log_psi_apply)(params, positions)
        return jax.flatten_util.ravel_pytree(log_grads)[0]

    batch_raveled_log_psi_grad = jax.vmap(raveled_log_psi_grad, in_axes=(None, 0))
   
    def raveled_log_psi_grad_x(params: P, positions: Array) -> Array:
        log_grads_x= jax.grad(log_psi_apply, argnums=1)(params, positions)
        return jnp.reshape(log_grads_x,(-1,))

    batch_raveled_log_psi_grad_x = jax.vmap(raveled_log_psi_grad_x, in_axes=(None, 0))
    
    
    def energy_data_val_no_key(params: P, positions: Array): #Here positions contains only one sample.
        
        #positions=jnp.expand_dims( positions, axis=0)
        out=energy_data_val(params, positions, jax.random.PRNGKey(0))        
        return out
    
    #batch_raveled_energy_data_val_no_key=jax.vmap(energy_data_val_no_key, in_axes=(None, 0))   
    
    def raveled_grad_energy_x(params: P, positions: Array):            #Here positions contains only one sample, output a vector
        def energy_data(positions_flattened):
            reshaped_positions = jnp.reshape(positions_flattened, positions.shape)
            return energy_data_val_no_key(params, reshaped_positions)
            
        grad_fn = jax.grad(energy_data)
        pos_flattened=jnp.reshape(positions, (-1,))
        return  grad_fn(pos_flattened)
    
    batch_raveled_grad_energy_x=jax.vmap(raveled_grad_energy_x, in_axes=(None, 0))  
    #batch_raveled_tanh_combine=jax.vmap(tanh_negative_grad_local_energy_and_grad_tanh_negative_grad_local_energy, in_axes=(None,None, None,0))  
    
    
    def get_w2_kfac_loss(params: P, positions: Array, V: Array): #V is the updated and clipped velocity
        V=lax.stop_gradient(V)
        return 2*jnp.sum(V*raveled_log_psi_grad_x(params, positions) )
        
        
        
        
    def raveled_W2_direction(params: P, positions: Array, V: Array):
        dV=jnp.clip(raveled_grad_energy_x(params, positions),-1.0,1.0)
        V=lax.stop_gradient(V+dV)
        def f(params, positions):
            return 2*jnp.sum(V*raveled_log_psi_grad_x(params, positions) )
        grad=jax.grad(f)(params, positions)
        return jax.flatten_util.ravel_pytree(grad)[0]
    
    
    batch_raveled_W2_direction=jax.vmap( raveled_W2_direction, in_axes=(None, 0))
       
    return None
    
#     def W2_update_fn(
#         params: P,
#         prev_eta:P, 
#         positions: Array,
#         kfac_optimizer,
#         sqrt_s,
#         alpha,
#         mu=0.99,
#         damping=0.1,
#         dt=1e-3,
#         key = jax.random.PRNGKey(0),
#     ) :
#    # -> Tuple[Array, P]:
       
#         nchains = positions.shape[0]
#         log_psi_grads = batch_raveled_log_psi_grad(params, positions) / jnp.sqrt(nchains)
#         Ohat = log_psi_grads - jnp.mean(log_psi_grads, axis=0, keepdims=True)
#         Ohat = 2*Ohat  #2*grad, grad log psi^2
        
        
#         d = jnp.mean(batch_raveled_W2_direction(params, positions), axis=0)
        
#         prev_eta_array,unravel_fn=jax.flatten_util.ravel_pytree(prev_eta)
       
        
#         next_positions = batch - dt*de
#         next_positions = lax.stop_gradient(next_positions)
#         return unravel_fn(new_eta),next_positions

#     return W2_update_fn


def get_batch_local_energy(energy_data_val):
    def energy_data_val_no_key(params: P, positions: Array): #Here positions contains only one sample.
        out=energy_data_val(params, positions, jax.random.PRNGKey(0))        
        return out
    return jax.vmap(energy_data_val_no_key, in_axes=(None, 0)) 
    
    
    
def get_batch_raveled_gradient_local_energy(energy_data_val):
    def energy_data_val_no_key(params: P, positions: Array): #Here positions contains only one sample.
        
        #positions=jnp.expand_dims( positions, axis=0)
        out=energy_data_val(params, positions, jax.random.PRNGKey(0))        
        return out
    
    def raveled_grad_energy_x(params: P, positions: Array):            #Here positions contains only one sample, output a vector
        def energy_data(positions_flattened):
            reshaped_positions = jnp.reshape(positions_flattened, positions.shape)
            return energy_data_val_no_key(params, reshaped_positions)
            
        grad_fn = jax.grad(energy_data)
        pos_flattened=jnp.reshape(positions, (-1,))
        return  grad_fn(pos_flattened)
    
    batch_raveled_grad_energy_x=jax.vmap(raveled_grad_energy_x, in_axes=(None, 0)) 
    return batch_raveled_grad_energy_x


def get_batch_gradient_local_energy(energy_data_val):
    def energy_data_val_no_key(params: P, positions: Array): #Here positions contains only one sample.
        out=energy_data_val(params, positions, jax.random.PRNGKey(0))        
        return out
    
    grad_fn=jax.grad(energy_data_val_no_key, argnums=1)
    
    
    batch_grad_energy_x=jax.vmap(grad_fn, in_axes=(None, 0)) 
    return batch_grad_energy_x



def get_statistics_from_local_energy(
    local_energies: Array, nchains: int, nan_safe: bool = True
) -> Tuple[Array, Array]:
    """Collectively reduce local energies to an average energy and variance.

    Args:
        local_energies (Array): local energies of shape (nchains,), possibly
            distributed across multiple devices via utils.distribute.pmap.
        nchains (int): total number of chains across all devices, used to compute a
            sample variance estimate of the local energy
        nan_safe (bool, optional): flag which controls if jnp.nanmean is used instead of
            jnp.mean. Can be set to False when debugging if trying to find the source of
            unexpected nans. Defaults to True.

    Returns:
        (chex.Numeric, chex.Numeric): local energy average, local energy (sample)
        variance
    """
    # TODO(Jeffmin) might be worth investigating the numerical stability of the XLA
    # compiled version of these two computations, since the quality of the gradients
    # is fairly crucial to the success of the algorithm
    if nan_safe:
        allreduce_mean = utils.distribute.nanmean_all_local_devices
    else:
        allreduce_mean = utils.distribute.mean_all_local_devices
    energy = allreduce_mean(local_energies)
    variance = (
        allreduce_mean(jnp.square(local_energies - energy)) * nchains / (nchains - 1)
    )  # adjust by n / (n - 1) to get an unbiased estimator
    return energy, variance


def get_clipped_energies_and_aux_data(
    local_energies_noclip: Array,
    nchains: int,
    clipping_fn ,
    nan_safe: bool,
):
    """Clip local energies if requested and return auxiliary data."""
    if clipping_fn is not None:
        # For the unclipped metrics, which are not used in the gradient, don't
        # do these in a nan-safe way. This makes nans more visible and makes sure
        # nans checkpointing will work properly.
        energy_noclip, variance_noclip = get_statistics_from_local_energy(
            local_energies_noclip, nchains, nan_safe=False
        )

        local_energies = clipping_fn(local_energies_noclip, energy_noclip)
        energy, variance = get_statistics_from_local_energy(
            local_energies, nchains, nan_safe=nan_safe
        )
    else:
        local_energies = local_energies_noclip
        energy, variance = get_statistics_from_local_energy(
            local_energies_noclip, nchains, nan_safe=nan_safe
        )

        # Even though there's no clipping function, still record noclip metrics
        # without nan-safety so that checkpointing epochs with nans can be
        # supported.
        energy_noclip, variance_noclip = get_statistics_from_local_energy(
            local_energies_noclip, nchains, nan_safe=False
        )
    aux_data = dict(
        energy=energy,
        variance=variance,
        local_energies_noclip=local_energies_noclip,
        energy_noclip=energy_noclip,
        variance_noclip=variance_noclip,
        centered_local_energies=local_energies - energy,
    )
    return local_energies, aux_data

def get_w2_kfac_loss(
    log_psi_apply: ModelApply[P],
    energy_data_val: ModelApply[P],
    clipping_fn
):
    """
    Get w2_kfac_loss function.

    Args:
        log_psi_apply (Callable): computes log|psi(x)|, where the signature of this
            function is (params, x) -> log|psi(x)|
    Returns:
        Callable: W2 loss for kfac optimizer
    """

#     raveled_log_grads_x_fn= jax.grad(log_psi_apply, argnums=1)
    
#     batch_raveled_log_grads_x=jax.vmap(raveled_log_psi_grad_x_fn, in_axes=(None, 0))
   
    def dir_grad(params: P, positions: Array, v: Array) -> Array:
         
        log_grads_x=jax.grad(log_psi_apply, argnums=1)(params, positions)
        
       
        return -2*jnp.sum(log_grads_x*v)
        #return jnp.reshape(log_grads_x,(-1,)) 

     
    batch_raveled_loss=jax.vmap(dir_grad,in_axes=(None, 0, 0)) 
    
#     @jax.custom_jvp
    def w2_kfac_loss( params,key, batch): #V is the updated and clipped velocity, tensor
        

        mcmc_positions,positions, V = batch
        V=lax.stop_gradient(V)
                
        
        
        per_sample_loss=batch_raveled_loss(params, positions, V)
        
        
        log_psi = log_psi_apply(params, positions)
        kfac_jax.register_normal_predictive_distribution(2*log_psi[:, None])
        
        
        w2_loss=jnp.mean(per_sample_loss)
        
        _,aux_data=get_clipped_energies_and_aux_data(
            local_energies_noclip=get_batch_local_energy(energy_data_val)(params,mcmc_positions),
            nchains=positions.shape[0],
            clipping_fn=clipping_fn,
            nan_safe=False,
            )
        
        return w2_loss, aux_data    
    
#     @w2_kfac_loss.defjvp
#     def w2_kfac_loss_jvp(primals, tangents):
#         #params,  positions, V = primals
#         params, key, batch = primals
        
#         positions =batch[1]
        
        
#         log_psi = log_psi_apply(params, positions)
#         kfac_jax.register_normal_predictive_distribution(2*log_psi[:, None])
        
#         loss,aux_data = w2_kfac_loss(params, key, batch)  #batch=(mcmc_positions,positions,lax.stop_gradient(V))
    

#         score =jax.vmap(jax.grad(log_psi_apply, argnums=1), in_axes=(None, 0))

#         score_primal, score_tangent = jax.jvp(score, (params, positions), (tangents[0], tangents[2][1]))
    
#         score_norm = lax.stop_gradient(jnp.linalg.norm(score_primal, axis=(-2, -1), keepdims=True))
#         median = jnp.median(lax.all_gather(score_norm, PMAP_AXIS_NAME))
#         deviation = jnp.mean(jnp.abs(score_norm - median))
#         mask = score_norm < (median + 5*deviation)
    
#         primals_out = loss,aux_data

#         V=batch[2]
#         q_tangent_out = -2*(V*score_tangent*mask).sum(1)*(len(mask)/mask.sum())


#         tangents_out = (q_tangent_out.mean(),aux_data)
#         return primals_out, tangents_out
        
    return   w2_kfac_loss
        
   
    



def _get_dt_schedule(
    optimizer_config: ConfigDict,
) -> LearningRateSchedule:
    if optimizer_config.dt_schedule_type == "constant":

        def dt_schedule(t):
            return optimizer_config.dt

    elif optimizer_config.dt_schedule_type == "inverse_time":

        def dt_schedule(t):
            return optimizer_config.dt / (
                1.0 + optimizer_config.dt_decay_rate * t
            )
    elif optimizer_config.dt_schedule_type == "inverse_time_lower_bound":

        def dt_schedule(t):
            return jnp.maximum(optimizer_config.dt / (1.0 + optimizer_config.dt_decay_rate * t),optimizer_config.dt_lower_bound)
    elif optimizer_config.dt_schedule_type == "sqrt_inverse_time":

        def dt_schedule(t):
            return optimizer_config.dt/ jnp.sqrt((
                1.0 + optimizer_config.dt_decay_rate * t
            ))
    elif optimizer_config.dt_schedule_type == "exponetial_decay":
        
        def dt_schedule(t):
            power = jnp.array(t / optimizer_config.T, dtype=int)
            return optimizer_config.dt *(
                optimizer_config.dt_decay_rate
            )**power
    else:
        raise ValueError(
            "dt schedule type not supported; {} was requested".format(
                optimizer_config.dt
            )
        )

    return dt_schedule


def _get_lrV_schedule(
    optimizer_config: ConfigDict,
) -> LearningRateSchedule:
    if optimizer_config.lrV_schedule_type == "constant":

        def lrV_schedule(t):
            return optimizer_config.lrV

    elif optimizer_config.lrV_schedule_type == "inverse_time":

        def lrV_schedule(t):
            return optimizer_config.lrV / (
                1.0 + optimizer_config.lrV_decay_rate * t
            )
    elif optimizer_config.lrV_schedule_type == "inverse_time_lower_bound":

        def lrV_schedule(t):
            return jnp.maximum(optimizer_config.lrV / (1.0 + optimizer_config.lrV_decay_rate * t),optimizer_config.lrV_lower_bound)
    elif optimizer_config.lrV_schedule_type == "sqrt_inverse_time":

        def lrV_schedule(t):
            return optimizer_config.lrV/ jnp.sqrt((
                1.0 + optimizer_config.lrV_decay_rate * t
            ))
    elif optimizer_config.lrV_schedule_type == "exponetial_decay":
        
        def lrV_schedule(t):
            power = jnp.array(t / optimizer_config.T, dtype=int)
            return optimizer_config.lrV *(
                optimizer_config.lrV_decay_rate
            )**power
    else:
        raise ValueError(
            "lrV schedule type not supported; {} was requested".format(
                optimizer_config.lrV
            )
        )

    return lrV_schedule



def constrain_norm_tensor(
    tensor,
    norm_constraint: chex.Numeric = 0.001,
):
    """Euclidean norm constraint."""
    sq_norm_scaled = jnp.sum(tensor**2)

   
    sq_norm_scaled = utils.distribute.pmean_if_pmap(sq_norm_scaled)
    
    norm_scale_factor = jnp.sqrt(norm_constraint / sq_norm_scaled)
    coefficient = jnp.minimum(norm_scale_factor, 1)
    constrained = tensor*coefficient

    return constrained


def constrain_norm(
    grad: P,
    norm_constraint: chex.Numeric = 0.001,
) -> P:
    """Euclidean norm constraint."""
    sq_norm_scaled_grads = tree_inner_product(grad, grad)

    # Sync the norms here, see:
    # https://github.com/deepmind/deepmind-research/blob/30799687edb1abca4953aec507be87ebe63e432d/kfac_ferminet_alpha/optimizer.py#L585
    sq_norm_scaled_grads = utils.distribute.pmean_if_pmap(sq_norm_scaled_grads)
    
    norm_scale_factor = jnp.sqrt(norm_constraint / sq_norm_scaled_grads)
    coefficient = jnp.minimum(norm_scale_factor, 1)
    constrained_grads = multiply_tree_by_scalar(grad, coefficient)

    return constrained_grads

def clip(vector, threshold):
    # 计算范数
    norm = jnp.linalg.norm(vector)
    
    
    def handle_inf(vector):
        return jnp.zeros_like(vector)
    
    def handle_large_norm(vector):
        return (vector / norm) * threshold
    
     
    def handle_normal(vector):
        return vector
    def normal_clip(vector):
        return lax.cond(
            norm > threshold,   
             handle_large_norm,   
             handle_normal,   
             vector
        )
     
    result = lax.cond(
        jnp.logical_or(jnp.isinf(norm), jnp.isnan(norm)),  
        handle_inf,  
        normal_clip,
        vector)
    
    return result

def nan_clip(vector):
     
    norm = jnp.linalg.norm(vector)
    
    
    def handle_inf(vector):
        return jnp.zeros_like(vector)
    
   
     
    def handle_normal(vector):
        return vector

    
    result = lax.cond(
        jnp.logical_or(jnp.isinf(norm), jnp.isnan(norm)),  
        handle_inf,   
        handle_normal,
        vector)
    
    return result
