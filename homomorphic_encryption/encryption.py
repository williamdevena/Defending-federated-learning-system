"""
This file contains functions used to implement homomorphic encryption.
"""

import tenseal as ts


def encrypt_parameters(param):
    """
    Returns encrypted parameters using Homomorphic Encryption.

    Args:
        - param (np.ndarray): parameters of a model

    Returns:
        - enc_param (np.ndarray): encrypted parameters of the model
    """
    # Setup TenSEAL context
    context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[60, 40, 40, 60]
            )
    context.generate_galois_keys()
    context.global_scale = 2**40
    enc_params = ts.ckks_vector(context, param)

    return enc_params


def main():
    pass


if __name__=="__main__":
    main()