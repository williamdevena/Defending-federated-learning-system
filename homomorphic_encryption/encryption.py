"""
This file contains functions used to implement homomorphic encryption.
"""

import logging
import pickle
import random

import numpy as np
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
    # param = np.array([
    #                 [.3,2.3,3,4,5],
    #                 [1,2,3,4,5]
    #                   ])
    # enc_param = encrypt_parameters(param=param.flatten())

    # enc_result = enc_param*0.5

    # file_name = 'enc_param.pkl'
    # with open(file_name, 'wb') as file:
    #     pickle.dump(enc_param, file)


    # print(enc_result.decrypt())



    #print(np.array(enc_param.decrypt()).reshape((2,5)))

    #print((enc_param+[1,1,1,1,1]).decrypt())



    # # Setup TenSEAL context
    # context = ts.context(
    #             ts.SCHEME_TYPE.CKKS,
    #             poly_modulus_degree=8192,
    #             coeff_mod_bit_sizes=[60, 40, 40, 60]
    #         )
    # context.generate_galois_keys()
    # context.global_scale = 2**40

    # v1 = [0, 1, 2, 3, 4]
    # v2 = [4, 3, 2, 1, 0]

    # # encrypted vectors
    # enc_v1 = ts.ckks_vector(context, v1)
    # enc_v2 = ts.ckks_vector(context, v2)
    # print(enc_v1, enc_v2)

    # result = enc_v1 + enc_v2
    # result.decrypt() # ~ [4, 4, 4, 4, 4]

    # result = enc_v1.dot(enc_v2)
    # result.decrypt() # ~ [10]

    # matrix = [
    # [73, 0.5, 8],
    # [81, -5, 66],
    # [-100, -78, -2],
    # [0, 9, 17],
    # [69, 11 , 10],
    # ]
    # result = enc_v1.matmul(matrix)
    # result.decrypt() # ~ [157, -90, 153]

    return True


if __name__=="__main__":
    main()