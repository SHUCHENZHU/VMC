"""Kinetic energy terms."""
from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax
import vmcnet.physics as physics
from vmcnet.utils.typing import Array, P, ModelApply


def create_laplacian_kinetic_energy(
    log_psi_apply: Callable[[P, Array], Array],
) -> ModelApply[P]:
    """Create the local kinetic energy fn (params, x) -> -0.5 (nabla^2 psi(x) / psi(x)).

    Args:
        log_psi_apply (Callable): a function which computes log|psi(x)| for single
            inputs x. It is okay for it to produce batch outputs on batches of x as long
            as it produces a single number for single x. Has the signature
            (params, single_x_in) -> log|psi(single_x_in)|

    Returns:
        Callable: function which computes the local kinetic energy for continuous
        problems (as opposed to discrete/lattice problems), i.e. -0.5 nabla^2 psi / psi.
        Evaluates on only a single configuration so must be externally vmapped
        to be applied to a batch of walkers.
    """
    grad_log_psi_apply = jax.grad(log_psi_apply, argnums=1)

    def kinetic_energy_fn(params: P, x: Array) -> Array:
        return -0.5 * physics.core.laplacian_psi_over_psi(grad_log_psi_apply, params, x)

    return kinetic_energy_fn






# def local_kinetic_energy(f):    #f is log_psi_apply
#     """Creates a function to for the local kinetic energy, -1/2 \nabla^2 ln|f|.

#   Args:
#     f: Callable with signature f(params, data), where params is the set of
#       (model) parameters of the (wave)function and data is the configurations to
#       evaluate f at, and returns the values of the log magnitude of the
#       wavefunction at those configurations.

#   Returns:
#     Callable with signature lapl(params, data), which evaluates the local
#     kinetic energy, -1/2f \nabla^2 f = -1/2 (\nabla^2 log|f| +
#     (\nabla log|f|)^2).
#     """
#     x_shape = x.shape
#     flat_x = jnp.reshape(x, (-1,))
#     n = flat_x.shape[0]
#     def flattened_grad_log_psi_of_flat_x(flat_x_in):
#         """Flattened input to flattened output version of grad_log_psi."""
#         grad_log_psi_out = grad_log_psi_apply(params, jnp.reshape(flat_x_in, x_shape))
#         return jnp.reshape(grad_log_psi_out, (-1,))
    
#     eye = jnp.eye(n)
    

#     def _body_fun(i, val):
#         primal, tangent = jax.jvp(flattened_grad_log_psi_of_flat_x, (flat_x,), (eye[i],))
#         return val + primal[i]**2 + tangent[i]
#     return -0.5 * lax.fori_loop(0, n, _body_fun, 0.0)



# def potential_energy(r_ae, r_ee, atoms, charges):
#     """Returns the potential energy for this electron configuration.

#   Args:
#     r_ae: Shape (nelectrons, natoms). r_ae[i, j] gives the distance between
#       electron i and atom j.
#     r_ee: Shape (neletrons, nelectrons, :). r_ee[i,j,0] gives the distance
#       between electrons i and j. Other elements in the final axes are not
#       required.
#     atoms: Shape (natoms, ndim). Positions of the atoms.
#     charges: Shape (natoms). Nuclear charges of the atoms.
#     """
#     v_ee = jnp.sum(jnp.triu(1 / r_ee[..., 0], k=1))
#     v_ae = -jnp.sum(charges / r_ae[..., 0])  # pylint: disable=invalid-unary-operand-type
#     r_aa = jnp.linalg.norm(atoms[None, ...] - atoms[:, None], axis=-1)
#     v_aa = jnp.sum(
#     jnp.triu((charges[None, ...] * charges[..., None]) / r_aa, k=1))
#     return v_ee + v_ae + v_aa


# def get_local_energy(f, atoms, charges):
#     """Creates function to evaluate the local energy.

#   Args:
#     f: Callable with signature f(data, params) which returns the log magnitude
#       of the wavefunction given parameters params and configurations data.
#     atoms: Shape (natoms, ndim). Positions of the atoms.
#     charges: Shape (natoms). Nuclear charges of the atoms.

#   Returns:
#     Callable with signature e_l(params, data) which evaluates the local energy
#     of the wavefunction given the parameters params and a single MCMC
#     configuration in data.
#     """
#     ke = local_kinetic_energy(f)

#     def _e_l(params, x):
#         """Returns the total energy.

#     Args:
#       params: network parameters.
#       x: MCMC configuration.
#         """
#         _, _, r_ae, r_ee = networks.construct_input_features(x, atoms)
#         potential = potential_energy(r_ae, r_ee, atoms, charges)
#         kinetic = ke(params, x)
#         return potential + kinetic

#     return _e_l



def get_W2_direction(log_psi_apply):  #get function that computes 2*<nabla log psi,nabla local energy>+laplacian local energy
    grad_fn=jax.grad(local_kinetic_energy(log_psi_apply), argnums=1)
    def laplacian_fn(params,positions):
        laplacian=jnp.trace(    jax.jacrev(grad_fn, argnums=1)(params,positions)     )  
        return laplacian
    def fn(params,positions):
        result=2*jnp.dot(jax.grad(log_psi_apply, argnums=1)(params,positions), grad_fn(params,positions) )+laplacian_fn(params,positions)
        return result
    return fn