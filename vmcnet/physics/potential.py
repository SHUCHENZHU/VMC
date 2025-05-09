"""Potential energy terms."""
from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp

from vmcnet.utils.typing import Array, ArrayLike, ModelApply, ModelParams


def compute_displacements(x: ArrayLike, y: ArrayLike) -> Array:
    """Compute the pairwise displacements between x and y in the second-to-last dim.

    Args:
        x (Array): array of shape (..., n_x, d)
        y (Array): array of shape (..., n_y, d)

    Returns:
        Array: pairwise displacements (x_i - y_j), with shape (..., n_x, n_y, d)
    """
    return jnp.expand_dims(x, axis=-2) - jnp.expand_dims(y, axis=-3)


# NOTE: the custom VJP on this method returns 0.0 for the gradient when the norm is
# zero, even though technically the gradient is undefined in this case. This is
# currently necessary to ensure that the EE potential energy doesn't give nans when
# using the IBP formulation.
# TODO (ggoldsh): rewrite the EE term of the IBP method to avoid this issue.

@jax.custom_vjp
def compute_soft_norm_inv(
    displacements: ArrayLike, softening_term: chex.Scalar = 0.0
) -> Array:
    """Compute an (optionally softened) norm, sqrt((sum_i x_i^2) + softening_term^2).

    Args:
        displacements (Array): array of shape (..., d)
        softening_term (chex.Scalar, optional): this amount squared is added to
            sum_i x_i^2 before taking the sqrt. The smaller this term, the closer the
            derivative gets to a step function (but the derivative is continuous except
            for for softening term exactly equal to zero!). When zero, gives the usual
            vector 2-norm. Defaults to 0.0.

    Returns:
        Array: array with shape displacements.shape[:-1]
    """
    #softening_term=1e-6
    result=1.0/ (jnp.sqrt(
            jnp.sum(jnp.square(displacements), axis=-1) + jnp.square(softening_term))
        )
    return result
    


def _soft_norm_inv_forward(displacements, softening_term):
    norm_inv = compute_soft_norm_inv(displacements, softening_term)
    return (
        norm_inv,
        (
            norm_inv,
            displacements,
            softening_term,
        ),
    )


def _soft_norm_inv_bwd(res, g):
    (norm_inv, displacements, softening_term) = res
    expanded_norm_inv = jnp.expand_dims(norm_inv, axis=-1)
    
    # compute displacements 's gradient,keep the same shape, three order tensor
    
    tile_dims = (1, 1, displacements.shape[-1])
    #expand_norm_inv_matrix=jnp.tile(expanded_norm_inv,  tile_dims )
    expand_norm_inv_matrix=expanded_norm_inv
    
    displacements_grad = (
        jnp.expand_dims(g, -1)
        * jnp.where( jnp.isinf(expanded_norm_inv), 0.0, -displacements * (expand_norm_inv_matrix**3))
    )
    
     
    
     
    return displacements_grad, 0.0  


compute_soft_norm_inv.defvjp(_soft_norm_inv_forward, _soft_norm_inv_bwd)


def _get_ion_ion_info(
    ion_locations: ArrayLike, ion_charges: ArrayLike
) -> Tuple[Array, Array]:
    """Get pairwise ion-ion displacements and charge-charge products."""
    ion_ion_displacements = compute_displacements(ion_locations, ion_locations)
    charge_charge_prods = jnp.expand_dims(ion_charges, axis=-1) * ion_charges
    return ion_ion_displacements, charge_charge_prods


def create_electron_ion_coulomb_potential(
    ion_locations: Array,
    ion_charges: Array,
    strength: chex.Scalar = 1.0,
    softening_term: chex.Scalar = 0.0,
    nparticles: Optional[int] = None,
) -> ModelApply[ModelParams]:
    """Computes the total coulomb potential attraction between electron and ion.

    Args:
        ion_locations (Array): an (n, d) array of ion positions, where n is the
            number of ion positions and d is the dimension of the space they live in
        ion_charges (Array): an (n,) array of ion charges, in units of one
            elementary charge (the charge of one electron)
        strength (chex.Scalar, optional): amount to multiply the overall interaction by.
            Defaults to 1.0.
        softening_term (chex.Scalar, optional): this amount squared is added to
            sum_i x_i^2 before taking the sqrt in the norm calculation. When zero, the
            usual vector 2-norm is used to compute distance. Defaults to 0.0.
        nparticles (Optional): when specified, only the first nparticles particles are
            used to calculate the electron ion potential. Defaults to None.

    Returns:
        Callable: function which computes the potential energy due to the attraction
        between electrons and ion. Has the signature
        (params, electron_positions of shape (..., n_elec, d))
        -> array of potential energies of shape electron_positions.shape[:-2]
    """

    def potential_fn(params: ModelParams, x: Array) -> Array:
        del params
        multiplier = 1.0
        if nparticles is not None:
            multiplier = x.shape[-2] / nparticles
            x = x[..., :nparticles, :]

        electron_ion_displacements = compute_displacements(x, ion_locations)
        electron_ion_distances_inv = compute_soft_norm_inv(
            electron_ion_displacements, softening_term=softening_term
        )
        coulomb_attraction = ion_charges * electron_ion_distances_inv
        return -strength * jnp.sum(coulomb_attraction, axis=(-1, -2)) * multiplier

    return potential_fn


def create_electron_electron_coulomb_potential(
    strength: chex.Scalar = 1.0,
    softening_term: chex.Scalar = 0.0,
    nparticles: Optional[int] = None,
) -> ModelApply[ModelParams]:
    """Computes the total coulomb potential repulsion between pairs of electrons.

    Args:
        strength (chex.Scalar, optional): amount to multiply the overall interaction by.
            Defaults to 1.0.
        softening_term (chex.Scalar, optional): this amount squared is added to
            sum_i x_i^2 before taking the sqrt in the norm calculation. When zero, the
            usual vector 2-norm is used to compute distance. Defaults to 0.0.
        nparticles (int, Optional): when specified, the contribution of the first
            nparticles particles to the electron electron potential is returned. This
            means if i,j<=nparticles then the full coulomb repulsion 1/|r_i -r_j| is
            included; if i<=nparticles but j>nparticles then half the coulomb repulsion
            1/|r_i-r_j| is included, and if i,j>nparticles then the coulomb repulsion
            1/|r_i-r_j| is neglected. Defaults to None.

    Returns:
        Callable: function which computes the potential energy due to the repulsion
        between pairs of electrons. Has the signature
        (params, electron_positions of shape (..., n_elec, d))
        -> array of potential energies of shape electron_positions.shape[:-2]
    """

    def potential_fn(params: ModelParams, x: Array) -> Array:
        del params
        electron_electron_displacements = compute_displacements(x, x)
        electron_electron_distances_inv = compute_soft_norm_inv(
            electron_electron_displacements, softening_term=softening_term
        )
        if nparticles is None:
            ee_repulsion = jnp.triu(strength * electron_electron_distances_inv, k=1)
            return jnp.sum(ee_repulsion, axis=(-1, -2))
        else:
            multiplier = x.shape[-2] / nparticles
            double_ee_repulsion = jnp.triu(
                strength * electron_electron_distances_inv, k=1
            ) + jnp.tril(strength * electron_electron_distances_inv, k=-1)
            double_firstn_repulsion = double_ee_repulsion[..., :nparticles, :]
            return jnp.sum(double_firstn_repulsion, axis=(-1, -2)) * multiplier / 2

    return potential_fn


def create_ion_ion_coulomb_potential(
    ion_locations: Array, ion_charges: Array
) -> ModelApply[ModelParams]:
    """Computes the total coulomb potential repulsion between stationary ions.

    Args:
        ion_locations (Array): an (n, d) array of ion positions, where n is the
            number of ion positions and d is the dimension of the space they live in
        ion_charges (Array): an (n,) array of ion charges, in units of one
            elementary charge (the charge of one electron)

    Returns:
        Callable: function which computes the potential energy due to the attraction
        between electrons and ion. Has the signature
        (params, electron_positions of shape (..., n_elec, d))
        -> array of potential energies of shape electron_positions.shape[:-2]
    """
    ion_ion_displacements, charge_charge_prods = _get_ion_ion_info(
        ion_locations, ion_charges
    )
    ion_ion_distances_inv = compute_soft_norm_inv(ion_ion_displacements)
    constant_potential = jnp.sum(
        jnp.triu(charge_charge_prods * ion_ion_distances_inv, k=1), axis=(-1, -2)
    )

    def potential_fn(params: ModelParams, x: ArrayLike) -> Array:
        del params, x
        return constant_potential

    return potential_fn
