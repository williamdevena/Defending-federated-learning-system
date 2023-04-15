"""
This file contains functions used to simulate a federated learning system that uses homomorphic encryption.
"""

import logging
import random
import time

import encryption
import numpy as np


def clients_send_parameters(num_clients, shape_param):
    """
    Simulates a group of client in federated learning system.

    Args:
        - num_clients (int): number of clients.

    Returns:
        - list_enc_params (List[Tuple]): contains the encrypted parameters.
    """
    list_enc_params = []
    for id_client in range(num_clients):
        logging.info(f"Client {id_client} sending encripted parameters")
        param = np.random.rand(*shape_param)
        num_samples = random.randint(100, 1000)
        enc_param = encryption.encrypt_parameters(param=param.flatten())
        list_enc_params.append((id_client, enc_param, num_samples))

    return list_enc_params


def client_sets_new_parameters(weighted_enc_param):
    """
    Client sets new parameters.

    Args:
        weighted_enc_param (_type_): _description_
    """
    final_dec_param = np.array(weighted_enc_param[0].decrypt())
    for enc_param in weighted_enc_param:
        final_dec_param += np.array(enc_param.decrypt())

    return final_dec_param




def server_aggregates_parameters(list_enc_params):
    """
    Simulates the server in a federated learning system.

    Args:
        list_enc_params (List[Tuple]): contains the encrypted parameters

    Returns: None
    """
    time.sleep(1)
    logging.info("\nServer aggregating encripted parameters")
    time.sleep(1)
    total_samples = sum([num_samples for _, _, num_samples in list_enc_params])
    weights = [num_samples/total_samples for _, _, num_samples in list_enc_params]
    enc_params = [enc_param for _, enc_param, _ in list_enc_params]
    weighted_enc_params = [enc_param*weight for enc_param, weight in list(zip(enc_params, weights))]

    return weighted_enc_params


def simulate_federated_learning_system():
    """
    Simulates a federated learning system, with several client and one
    server.
    """
    list_enc_params = clients_send_parameters(10, [10, 10])
    weighted_enc_params = server_aggregates_parameters(list_enc_params=list_enc_params)
    final_dec_param = client_sets_new_parameters(weighted_enc_param=weighted_enc_params)
    logging.info(f"\nNew aggregated parameters:\n{final_dec_param}")




def main():
    ## LOGGING CONFIGURATION
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            #logging.FileHandler("project_log/assignment.log"),
            logging.StreamHandler()
        ]
    )
    logging.info("\n\nSIMULATION OF A FEDERATED LEARNING SYSTEM WITH HOMOMORPHIC ENCRYPTION\n\n")
    time.sleep(2)

    ## SIMULATION EXECUTION
    simulate_federated_learning_system()


if __name__=="__main__":
    main()