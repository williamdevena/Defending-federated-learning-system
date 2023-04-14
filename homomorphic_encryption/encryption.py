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





def simulate_clients(num_clients, shape_param):
    """
    Simulates a group of client in federated learning system.

    Args:
        - num_clients (int): number of clients.

    Returns:
        - list_enc_params (List[Tuple]): contains the encrypted parameters.
    """
    list_enc_params = []
    for id_client in range(num_clients):
        param = np.random.rand(shape_param)
        num_samples = random.randint(100, 1000)
        enc_param = encrypt_parameters(param=param.flatten())
        list_enc_params.append((id_client, enc_param, num_samples))

    return list_enc_params




def simulate_server(list_enc_params):
    """
    Simulates the server in a federated learning system.

    Args:
        list_enc_params (List[Tuple]): contains the encrypted parameters

    Returns: None
    """
    total_samples = sum([num_samples for _, _, num_samples in list_enc_params])
    weights = [num_samples/total_samples for _, _, num_samples in list_enc_params]
    enc_params = [enc_param for _, enc_param, _ in list_enc_params]
    weighted_enc_params = [enc_param*weight for enc_param, weight in list(zip(enc_params, weights))]

    print(total_samples, weights)

    return weighted_enc_params






def main():


    list_enc_params = simulate_clients(3, (4))
    weighted_enc_params = simulate_server(list_enc_params=list_enc_params)

    for enc_param in weighted_enc_params:
        print(enc_param.decrypt())



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


if __name__=="__main__":
    main()