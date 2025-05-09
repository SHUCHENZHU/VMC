"""RACC routine."""
import jax
import jax.flatten_util
import jax.numpy as jnp
from jax import jvp, grad
from vmcnet.utils.typing import Array, ModelApply, P, Tuple

import chex
from vmcnet.utils.pytree_helpers import (
    multiply_tree_by_scalar,
    tree_inner_product,
)
from vmcnet import utils
from jax import tree_util

def get_racc_cotangent_update_fn(
    log_psi_apply: ModelApply[P],
    damping: chex.Scalar = 0.001
):
    """
    Get the racc update function.

    Args:
        log_psi_apply (Callable): computes log|psi(x)|, where the signature of this
            function is (params, x) -> log|psi(x)|
        damping (float): damping parameter
        mu (float): racc-specific regularization

    Returns:
        Callable: racc update function. Has the signature
        (centered_energies, params, prev_grad, positions) -> new_grad
    """

    def raveled_log_psi_grad(params: P, positions: Array) -> Array:
        log_grads = jax.grad(log_psi_apply)(params, positions)
        return jax.flatten_util.ravel_pytree(log_grads)[0]

    batch_raveled_log_psi_grad = jax.vmap(raveled_log_psi_grad, in_axes=(None, 0))
   
     # EXAMPLE of HVP
    def hvp(f, primals, tangents):
#         print("primals:", primals)
#         print("tangents:", tangents)
        return jvp(grad(f), (primals,), (tangents,))[1]
   
    def raveled_log_psi_hvp(params: P, positions: Array, tangent_params: P) -> Array:
        def log_psi_apply_positions(params):
            return log_psi_apply(params,positions)
        hvp_log_psi =hvp(log_psi_apply_positions, params,tangent_params ) #默认计算第一个导数，第二个位置不影响
        return jax.flatten_util.ravel_pytree(hvp_log_psi)[0]
    
    batch_raveled_log_psi_hvp = jax.vmap(raveled_log_psi_hvp, in_axes=(None, 0,None))
  
    
    
    def racc_cotangent_update_fn(
        centered_energies: P,
        params: P,
        prev_eta:P, 
        prev_phi: Array,
        positions: Array,
        sqrt_s,
        alpha,
        correction_rate
    ) :
   # -> Tuple[Array, P]:
        nchains = positions.shape[0]

        
        
        log_psi_grads = batch_raveled_log_psi_grad(params, positions) / jnp.sqrt(
            nchains
        )
        Ohat = log_psi_grads - jnp.mean(log_psi_grads, axis=0, keepdims=True)
        Ohat = 2*Ohat  #2*grad, grad log psi^2
        T =  Ohat @ Ohat.T
        ones = jnp.ones((nchains, 1))
        T_reg = T + ones @ ones.T / nchains + damping * jnp.eye(nchains)

        epsilon_bar = centered_energies / jnp.sqrt(nchains)
        loss_grad=Ohat.T @ epsilon_bar
  
        
        prev_eta_array = jax.flatten_util.ravel_pytree(prev_eta)[0]
        unravel_fn = jax.flatten_util.ravel_pytree(params)[1]
        A=4*batch_raveled_log_psi_hvp(params, positions,unravel_fn(prev_eta_array)).T@(Ohat@prev_eta_array)/ jnp.sqrt( nchains)
        
        B=Ohat.T @ ( (Ohat@prev_eta_array)** 2)* jnp.sqrt(nchains)
        phi=(1- alpha*sqrt_s) *prev_phi+sqrt_s*loss_grad+0.5*sqrt_s*(A+B)*correction_rate
        eta = Ohat.T @ jax.scipy.linalg.solve(
           damping * jnp.eye(nchains)+T_reg@T_reg, Ohat @phi, assume_a="pos"
        )
      
        def true_fn(phi,new_eta):
            print('Ascent direction')
            return phi*0.0,new_eta * 0.0
        def false_fn(phi,new_eta):
            return phi,new_eta
        
        phi, eta = jax.lax.cond(jnp.dot(epsilon_bar, Ohat@eta)/ (jnp.linalg.norm(epsilon_bar)*jnp.linalg.norm(Ohat@eta)) < -0.0, true_fn, false_fn, phi,eta)
        
        

        return phi, unravel_fn(eta)
    
    
    
    
    
    
    

    return racc_cotangent_update_fn


def constrain_norm(
    grad: P,
    phi:Array,
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
    phi_clip= phi*coefficient
    return constrained_grads,phi_clip


