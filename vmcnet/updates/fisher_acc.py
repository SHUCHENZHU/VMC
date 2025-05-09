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

def subtract_random_mean(key, M,p):
    num_rows = M.shape[0]   
    random_matrix = jax.random.bernoulli(key, p=p, shape=(num_rows, num_rows))
    random_matrix =random_matrix /jnp.sum(random_matrix, axis=1, keepdims=True)
    result = M-random_matrix@M
    return result

def clip_median(data):
    median = jnp.median(data)
    median = utils.distribute.pmean_if_pmap(median)
    deviation = jnp.mean(jnp.abs(data - median))
    return jnp.clip(data, median - 5*deviation, median + 5*deviation)

def get_batch_raveled_dir_derivative(log_psi_apply, d_pytree):
    
    def dir_derivative_fn(params, position):
     
        _, dir_derivative = jax.jvp(
        lambda params: log_psi_apply(params, position),  
        (params,),  
        (d_pytree,)  
        )
    
        return dir_derivative
    return jax.vmap(dir_derivative_fn, in_axes=(None, 0))

def clip_vector_norm(vector, max_norm):

    length = jnp.linalg.norm(vector)
    
    clipped_length = jnp.clip(length, a_min=0.0, a_max=max_norm)
    
    jax.lax.cond(
        length>max_norm,
        lambda _: None,#jax.debug.print('clip'),
        lambda _: None,
        operand=None
        )
        
    normalized_vector = vector / (length + 1e-8)
    
    clipped_vector = normalized_vector * clipped_length
    return clipped_vector




    




def get_fisher_acc_update_fn(
    log_psi_apply: ModelApply[P],
    local_energy_fn,
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
    

     
    
    def fisher_acc_update_fn(
        epoch_step,
        centered_energies: Array,
        params: P,
        prev_eta:P,
        prev_cot :Array,
        positions: Array,
        sqrt_s1,
        prev_s1, 
        sqrt_s2,
        alpha,
        beta,
        balance,  #1.0
        mu, #spring
        prev_eta_proj='id',
        restart=False,
        key = jax.random.PRNGKey(0),
    ) :
      
           
        prev_eta_array,unravel_fn=jax.flatten_util.ravel_pytree(prev_eta)
        params_array,_=jax.flatten_util.ravel_pytree(params)
                
        
       
        nchains = positions.shape[0]
        log_psi_grads = batch_raveled_log_psi_grad(params, positions) #do not divide nchains
        
        
        Ohat = log_psi_grads - jnp.mean(log_psi_grads, axis=0, keepdims=True)
        Ohat = 2*Ohat 
        ones = jnp.ones((nchains, 1))
        T_reg = Ohat@Ohat.T + ones @ ones.T  + damping * jnp.eye(nchains)
            

        
        
        
        
        def update_fn(_):  
            prev_params = unravel_fn(params_array + prev_s1 * prev_eta_array)
            
            dir_derivative_fn=get_batch_raveled_dir_derivative(log_psi_apply, prev_eta)
            
            prev_cot_updated=-2*dir_derivative_fn(prev_params,positions) 
            prev_cot_updated=prev_cot_updated-jnp.mean(prev_cot_updated) 
            del prev_params
            return prev_cot_updated

        def zero_update_fn(_):
            return jnp.zeros(nchains)
        
        def id_update_fn(_):
            return prev_cot
        
        
                
            
        prev_Phi = lax.cond(
            jnp.any(jnp.isnan(prev_cot)) ,
            zero_update_fn,
            update_fn,
            operand=None
            )
        prev_Phi=prev_Phi-jnp.mean(prev_Phi)
        
        beta = lax.cond(
            jnp.any(jnp.isnan(prev_cot)),
            lambda: jnp.array(0.0),  # set beta=0
            lambda: beta,
            )
            

        
        

        
        
        def energy_data_val_no_key(params: P, positions: Array): #Here positions contains only one sample.
            out=local_energy_fn(params, positions, jax.random.PRNGKey(0))        
            return out
        batch_raveled_local_energy=jax.vmap( energy_data_val_no_key, in_axes=(None, 0))
     
        centered_energies=centered_energies[:nchains]
        
        
        def Hess_damping_fn(_):
            prev_params = unravel_fn(params_array + prev_s1 * prev_eta_array)
            prev_local_energy=batch_raveled_local_energy(prev_params, positions)
            prev_local_energy=clip_median(prev_local_energy)
            centered_prev_local_energy=prev_local_energy -jnp.mean(prev_local_energy)
            del prev_params
            return centered_energies-centered_prev_local_energy, centered_prev_local_energy
        
        def zero_Hess_damping_fn(_):
            return jnp.zeros(nchains),  jnp.zeros(nchains)
        
        Hess_damping,centered_prev_local_energy = lax.cond(
            beta>1e-6 ,
            Hess_damping_fn,
            zero_Hess_damping_fn,
            operand=None
            )
        
        prev_Psi=prev_Phi+beta*centered_prev_local_energy  
        new_Phi=(1-alpha*sqrt_s2)* prev_Phi-sqrt_s2* 0.5*prev_Psi*prev_Phi-sqrt_s2*centered_energies- beta*Hess_damping
        
        new_Phi=clip_median(new_Phi)  # clip Phi, numerical stable
        new_Phi=new_Phi-jnp.mean(new_Phi) #centered , do not influence updates
       
        if prev_eta_proj=='Null_O':
            mu=mu
        elif prev_eta_proj=='id':
            mu=0.0
            
        prev_eta_array=clip_vector_norm(prev_eta_array, 1.0)  
        
        b=-new_Phi- mu*Ohat@ prev_eta_array
        x=Ohat.T @jax.scipy.linalg.solve(T_reg, b, assume_a="pos")
        new_eta=x+mu*prev_eta_array

        
        
        
        
        
        
        # nan test 
#         jax.lax.cond(
#         jnp.any(jnp.isnan(min_new_eta)),
#         lambda _: jax.debug.print('nan_min_new_eta'),
#         lambda _: None,
#         operand=None
#         )
        
        
        
        def true_fn(new_eta,new_cot):
            jax.debug.print('Ascent')
            return 0.0*new_eta, jnp.full_like(new_cot, jnp.nan)
        
        def true_fn1(new_eta,new_cot):
            jax.debug.print('NAN_eta')
            return 0.0*new_eta, jnp.full_like(new_cot, jnp.nan)
        def false_fn(new_eta,new_cot):
            return new_eta,new_cot
        
        
        
        #new_eta,new_Phi = lax.cond(jnp.any(jnp.isnan(new_eta)), true_fn1, false_fn, new_eta,new_Phi)
        
        
        if restart:
            d0=Ohat.T@centered_energies
            d=jnp.dot(new_eta,d0)/jnp.sqrt(jnp.dot(new_eta,new_eta)*jnp.dot(d0,d0))
            new_eta,new_Phi = lax.cond(d < -0.75, true_fn, false_fn, new_eta,new_Phi)
            
            
        

        return unravel_fn(new_eta),new_Phi 

    
    
    
    
    return fisher_acc_update_fn

def constrain_norm(
    grad: P,
    cot_vec: Array,
    cot_vec_threshold: chex.Numeric =1.0, #per element
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
    
    
    cot_vec = jax.lax.cond(
        sq_norm_scaled_grads > norm_constraint,
        lambda cot_vec: (jax.debug.print("Warning: Gradient norm exceeds the constraint: {}", sq_norm_scaled_grads),
                     jnp.full_like(cot_vec, jnp.nan))[1],
#         lambda cot_vec: jnp.full_like(cot_vec, jnp.nan),
        lambda cot_vec: cot_vec,
        cot_vec
    )
    return constrained_grads,cot_vec




