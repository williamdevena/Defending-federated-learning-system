import logging
import torch
import os
import flwr as fl
import numpy as np
from flwr.common import Metrics
    
    
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy


from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy.strategy import Strategy

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


class FedCustom(Strategy):
    """Configurable FedAvg strategy implementation."""

    def __init__(
        self,
        *,
        fraction_fit: float = 0.5,
        fraction_evaluate: float = 0.5,
        min_fit_clients: int = 5,
        min_evaluate_clients: int = 5,
        min_available_clients: int = 10,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        threshold_list: List[float] = None,
        back_index: int = 0
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.last_aggregated_parameter = None
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = weighted_average #evaluate_metrics_aggregation_fn
        self.back_index = back_index if back_index != 0 else 0
        if threshold_list is not None:
            self.threshold_list = threshold_list
        else:
            if os.path.isfile("thresholds.txt"):
                self.threshold_list = np.loadtxt("thresholds.txt")
            else:
                self.threshold_list = [[0.0] * 10, [0.0] * 10, [0.0] * 10]


    def __repr__(self) -> str:
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available
        clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients


    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
    
    def save_threshold_list(self, filename: str) -> None:
        np.savetxt("thresholds.txt", self.threshold_list)

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters


    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        
        if server_round == 1:
            self.back_index = 0
        
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [    (fit_res.parameters, fit_res.num_examples)    
                           for _, fit_res in results]
        
        print(self.threshold_list)
        # shape of weights_results (3,2)
        # len of num_examples = 50000
        
        diff_norm_list, del_index = [], []
        iter = 0
        print(self.back_index)
        upper_bound = self.threshold_list[0][server_round-1-self.back_index] + 6 * self.threshold_list[2][server_round-1-self.back_index]
        print(upper_bound)
        lower_bound = self.threshold_list[1][server_round-1-self.back_index] - 6 * self.threshold_list[2][server_round-1-self.back_index]
        print(lower_bound)
        std_threshold = self.threshold_list[2][server_round-1-self.back_index]
        for _, fit_res in results:
            cid = fit_res.metrics["cid"]
            
            # Check if difference between last aggregated parameter and client parameter
            # is less than a certain threshold before aggregating update
            if self.last_aggregated_parameter is not None:
                print(f"Checking the gradient on Cleint {cid}")
                client_parameter = parameters_to_ndarrays(weights_results[iter][0])
                last_parameter = parameters_to_ndarrays(self.last_aggregated_parameter)
                last_parameter = [np.array(elem) for elem in last_parameter]
                diff_norm = np.linalg.norm([np.linalg.norm(a - b) for a, b in zip(client_parameter, last_parameter)])
                print(f"The diff is {diff_norm}")
                
                if diff_norm >=  upper_bound or diff_norm <= lower_bound:
                    print(f"Client {cid} parameter diff norm {diff_norm} is bigger than {upper_bound} or smaller than {lower_bound}. Skipping client update.")
                    # return self.last_aggregated_parameter, {}
                    del_index.append(iter)
                else:
                    diff_norm_list.append(diff_norm)
                
                diff_norm_list.append(diff_norm)
                iter += 1
                
        # Delete those fake client data
        if self.last_aggregated_parameter is not None:
            if len(del_index) >= 3:
                if server_round != 1:
                    self.back_index += 1
                return self.last_aggregated_parameter, {}
            else:
                # Find the index of the first remaining element
                first_remaining_index = next(i for i in range(len(weights_results)) if i not in del_index)

                # Access the underlying NumPy arrays of the first remaining element
                first_remaining_weights = parameters_to_ndarrays(weights_results[first_remaining_index][0])
                # print(f"Row {len(first_remaining_weights[0])}, Column {len(first_remaining_weights)}")
                # Create a modified version with added noise
                modified_weights = [np.add(w, np.random.normal(-std_threshold, std_threshold, w.shape)) for w in first_remaining_weights]
                # print(f"Row {len(modified_weights[0])}, Column {len(modified_weights)}")

                # Replace unwanted elements with the modified version
                updated_weights_results = []
                for i, (weight, num_examples) in enumerate(weights_results):
                    if i in del_index:
                        updated_weights_results.append((ndarrays_to_parameters(modified_weights), num_examples))
                    else:
                        updated_weights_results.append((weight, num_examples))

                weights_results = tuple(updated_weights_results)

                    
        # Aggregate results
        weights_aggregated = aggregate([(parameters_to_ndarrays(weights), num_examples) for weights, num_examples in weights_results])
        parameters_aggregated = ndarrays_to_parameters(weights_aggregated)
        
        # Update threshold list (DO NOT UNCOMMENT IT UNTIL WE GET ATTECK AND DEFENSE DONE)       
        if self.last_aggregated_parameter is not None and len(del_index) == 0:
            
            print(f"np.mean(diff_norm_list) (This round) = {np.mean(diff_norm_list)}")
            print(f'std_value (This round) = {np.std(diff_norm_list)}')
            
            self.threshold_list[0][server_round-1] = np.max([np.mean(diff_norm_list), self.threshold_list[0][server_round-1]])
            print("Currently the max thresholdis: ", self.threshold_list[0])
            if self.threshold_list[1][server_round-1] == 0:
                self.threshold_list[1][server_round-1] = np.mean(diff_norm_list)
                print("Currently the min thresholdis: ", self.threshold_list[1])
            else:
                self.threshold_list[1][server_round-1] = np.min([np.mean(diff_norm_list), self.threshold_list[1][server_round-1]])
                print("Currently the min thresholdis: ", self.threshold_list[1])
                
            self.threshold_list[2][server_round-1] = np.max([np.std(diff_norm_list), self.threshold_list[2][server_round-1]])
            print("Currently the std: ", self.threshold_list[2])
            # print(f'Currnet threshold_list variable {self.threshold_list}')

        # Save last aggregated parameter for future comparison
        self.last_aggregated_parameter = parameters_aggregated

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")
            
        

        return parameters_aggregated, metrics_aggregated


    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}



# Define strategy
# strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)
strategy = FedCustom()

# Define the Reputation Score


# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)


strategy.save_threshold_list("thresholds.txt")