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

def Kernel(points, sample_num,kernel_type='gaussian',key=jax.random.PRNGKey(0)):
    n=points.shape[0]
    d=points.shape[1]
    if kernel_type=='gaussian':
        key, subkey = jax.random.split(key)
        
        indices = jax.random.choice(subkey, n, shape=(sample_num,), replace=False)
        sampled_points = points[indices, :]
        distance_squared_matrix = jnp.sum((sampled_points[:, jnp.newaxis] - points) ** 2, axis=2)
        h=jnp.median(distance_squared_matrix[ :,indices])   /(2*jnp.log(sample_num))
        #Kernel_mat = jnp.exp(-distance_squared_matrix/(2*h))   / jnp.sqrt((2 * jnp.pi*h) ** d)
        Kernel_mat = jnp.exp(-distance_squared_matrix/(2*h))  
        Kernel_mat=Kernel_mat/jnp.mean(Kernel_mat)
        
    if kernel_type=='quadratic_affine':
        
        Kernel_mat=jnp.hstack((points**2, points))
        Kernel_mat=Kernel_mat.T 

        
    return   Kernel_mat 



def get_racc_update_fn(
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
    def H_bilinear(f,x, v):
        def f_x_v(t):
            return f(tree_util.tree_map(lambda xi, vi: xi + t * vi, x, v))
        return jax.jacfwd(jax.grad(f_x_v))(0.0)   #是否要冻结维度？
    def raveled_log_psi_hvp(params: P, positions: Array, tangent_params: P) -> Array:
        def log_psi_apply_positions(params):
            return log_psi_apply(params,positions)
        hvp_log_psi =hvp(log_psi_apply_positions, params,tangent_params ) #默认计算第一个导数，第二个位置不影响
        return jax.flatten_util.ravel_pytree(hvp_log_psi)[0]
    def raveled_log_psi_H_bilinear(params: P, positions: Array, tangent_params: P) -> Array:
        def log_psi_apply_positions(params):
            return log_psi_apply(params,positions)
        H_bilinear_log_psi =H_bilinear(log_psi_apply_positions, params,tangent_params ) #默认计算第一个导数，第二个位置不影响
        return jax.flatten_util.ravel_pytree(H_bilinear_log_psi )[0] 
    batch_raveled_log_psi_hvp = jax.vmap(raveled_log_psi_hvp, in_axes=(None, 0,None))
    batch_raveled_log_psi_H_bilinear = jax.vmap(raveled_log_psi_H_bilinear, in_axes=(None, 0,None))
    
    def racc_update_fn(
        centered_energies: P,
        params: P,
        prev_eta:P, 
        positions: Array,
        sqrt_s,
        alpha,
        prev_eta_proj=False,
        correction_rate=1.0,
        restart=True,
        key = jax.random.PRNGKey(0),
    ) :
   # -> Tuple[Array, P]:
        nchains = positions.shape[0]

        
        
        log_psi_grads = batch_raveled_log_psi_grad(params, positions) / jnp.sqrt(nchains)
        Ohat = log_psi_grads - jnp.mean(log_psi_grads, axis=0, keepdims=True)
        Ohat = 2*Ohat  #2*grad, grad log psi^2
        #T =  Ohat @ Ohat.T
        
        epsilon_bar = centered_energies / jnp.sqrt(nchains)
        prev_eta_array,unravel_fn=jax.flatten_util.ravel_pytree(prev_eta)
        
        
        if isinstance(prev_eta_proj, ml_collections.config_dict.config_dict.ConfigDict) and prev_eta_proj['kernel']==True:
            print('kernel')
            sample_num=prev_eta_proj['subsample_num']
            K=Kernel(jnp.reshape(positions, (positions.shape[0], -1)), sample_num=sample_num,kernel_type=prev_eta_proj['kernel_type'],key=key)
            
            Ohat_K=K@Ohat
           
            T_reg=Ohat_K@ Ohat_K.T
            T_reg=T_reg+damping *jnp.diag(jnp.linalg.norm(T_reg, axis=1))
            epsilon_bar=K@epsilon_bar
            grad=Ohat_K.T @ jax.scipy.linalg.solve(T_reg,epsilon_bar , assume_a="pos")
            
           
            new_eta=(1- alpha*sqrt_s) *prev_eta_array+sqrt_s*grad
            return unravel_fn(new_eta)
        else:
            T=Ohat@Ohat.T
            ones = jnp.ones((nchains, 1))
            T_reg = T + ones @ ones.T / nchains + damping * jnp.eye(nchains)

        
        #loss_grad=Ohat.T @ epsilon_bar
        
  
        
        
        
        if isinstance(prev_eta_proj, ml_collections.config_dict.config_dict.ConfigDict) and prev_eta_proj['svd']==True:
            print('svd')

            topk=prev_eta_proj['svd_dim']
           
            eigvecs,eigvals  = jax.lax.linalg.eigh(T+ damping * jnp.eye(nchains))  
            idx = jnp.argsort(eigvals)[::-1][:topk]  # 选出前k个特征值的索引
            S_reduced = eigvals[idx]
            U_reduced = eigvecs[:, idx]
            
            #U, S, _= jnp.linalg.svd(T,hermitian=True)           
#             U_reduced = U[:, :topk] 
#             S_reduced = S[:topk]+ damping

            Gamma1_reduced=  jnp.diag(1.0/jnp.sqrt(S_reduced))  @  U_reduced.T@Ohat #Gamma 前topk行
           
            #proj_rgrad=Gamma1_reduced.T@ jnp.diag(1.0/jnp.sqrt(S_reduced)) @ (U_reduced.T@epsilon_bar)
            proj_rgrad=Ohat.T @ jax.scipy.linalg.solve(T_reg,epsilon_bar , assume_a="pos")
            proj_momentum=prev_eta_array-Gamma1_reduced.T@(Gamma1_reduced@prev_eta_array)
           
            new_eta=(1- alpha*sqrt_s) *proj_momentum+sqrt_s*proj_rgrad
            return unravel_fn(new_eta)
    
    
        #prev_eta_array = jax.flatten_util.ravel_pytree(prev_eta)[0]
        #unravel_fn = jax.flatten_util.ravel_pytree(params)[1]
        #A=4*batch_raveled_log_psi_hvp(params, positions,unravel_fn(prev_eta_array)).T@(Ohat@prev_eta_array)/ jnp.sqrt( nchains)
        A=2*batch_raveled_log_psi_H_bilinear(params, positions,prev_eta)/ jnp.sqrt(nchains)
        A=jnp.squeeze(A)
        B=( (Ohat@prev_eta_array)** 2)* jnp.sqrt(nchains)
        centered_A=A-jnp.mean(A, axis=0, keepdims=True)
        centered_B=B-jnp.mean(B, axis=0, keepdims=True)
        #phi=(1- alpha*sqrt_s) * prev_phi+sqrt_s*loss_grad+0.5*sqrt_s*(A+B)
        
        b=epsilon_bar+(0.5*centered_B+centered_A)*correction_rate
        new_eta = Ohat.T @ jax.scipy.linalg.solve(
            T_reg, b , assume_a="pos"
        )
        if prev_eta_proj=='Null_O':
            prev_eta_array=prev_eta_array-Ohat.T @ jax.scipy.linalg.solve(T_reg, Ohat@ prev_eta_array, assume_a="pos")
            new_eta=(1- alpha*sqrt_s) * prev_eta_array+sqrt_s*new_eta
        elif prev_eta_proj=='Im_O_T':
            prev_eta_array=Ohat.T @ jax.scipy.linalg.solve(T_reg, Ohat@ prev_eta_array, assume_a="pos")
            new_eta=(1- alpha*sqrt_s) * prev_eta_array+sqrt_s*new_eta
        elif prev_eta_proj=='id':
            new_eta=(1- alpha*sqrt_s) * prev_eta_array+sqrt_s*new_eta
        elif prev_eta_proj=='proj_grad':
            proj_ImO=Ohat.T @ jax.scipy.linalg.solve(T_reg, Ohat@ prev_eta_array, assume_a="pos")
            coefficient=jnp.dot(proj_ImO,new_eta)/jnp.dot(proj_ImO,proj_ImO)
            proj_NullO=coefficient*(prev_eta_array-proj_ImO)
            clip_proj_NullO=  nan_clip( proj_NullO) #clip( proj_NullO,1.0)
            rgrad_est=clip_proj_NullO+new_eta
            new_eta=(1- alpha*sqrt_s) * prev_eta_array+sqrt_s*rgrad_est
        elif  prev_eta_proj=='Im_O_T+Null_O':
            proj=prev_eta_array-Ohat.T @ jax.scipy.linalg.solve(T_reg, Ohat@ prev_eta_array, assume_a="pos")
            new_eta=(1- alpha*sqrt_s) * (0.5*prev_eta_array+0.5*proj)+sqrt_s*new_eta
        elif isinstance(prev_eta_proj, (int, float, jnp.integer, jnp.floating)):
            proj=prev_eta_array-Ohat.T @ jax.scipy.linalg.solve(T_reg, Ohat@ prev_eta_array, assume_a="pos")
            new_eta=(1- alpha*sqrt_s) * (prev_eta_proj*prev_eta_array+(1-prev_eta_proj)*proj)+sqrt_s*new_eta
         #new_eta=(1- alpha*sqrt_s) * prev_eta_array+new_eta
       
        
        def true_fn(new_eta):
            print('Asent',sqrt_s)
            return new_eta * 0.0
        def false_fn(new_eta):
            return new_eta
        if restart:
            new_eta = lax.cond(jnp.dot(epsilon_bar, Ohat@new_eta)/ (jnp.linalg.norm(epsilon_bar)*jnp.linalg.norm(Ohat@new_eta)) < -0.0, true_fn, false_fn, new_eta)

        return unravel_fn(new_eta)

    return racc_update_fn


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
    
    # 定义处理无穷大的函数
    def handle_inf(vector):
        return jnp.zeros_like(vector)
    
    # 定义处理大于阈值的函数
    def handle_large_norm(vector):
        return (vector / norm) * threshold
    
    # 定义处理正常情况下的函数
    def handle_normal(vector):
        return vector
    def normal_clip(vector):
        return lax.cond(
            norm > threshold,  # 检查 norm 是否大于阈值
             handle_large_norm,  # 如果大于阈值
             handle_normal,  # 否则
             vector
        )
    # 使用 lax.cond 处理无穷大的情况
    result = lax.cond(
        jnp.logical_or(jnp.isinf(norm), jnp.isnan(norm)),  # 检查 norm 是否为无穷大
        handle_inf,  # 如果是无穷大
        normal_clip,
        vector)
    
    return result

def nan_clip(vector):
    # 计算范数
    norm = jnp.linalg.norm(vector)
    
    # 定义处理无穷大的函数
    def handle_inf(vector):
        return jnp.zeros_like(vector)
    
   
    # 定义处理正常情况下的函数
    def handle_normal(vector):
        return vector

    # 使用 lax.cond 处理无穷大的情况
    result = lax.cond(
        jnp.logical_or(jnp.isinf(norm), jnp.isnan(norm)),  # 检查 norm 是否为无穷大
        handle_inf,  # 如果是无穷大
        handle_normal,
        vector)
    
    return result