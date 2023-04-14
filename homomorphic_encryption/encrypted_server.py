import inspect
from functools import reduce
from logging import WARNING
from typing import List, Tuple

import flwr as fl
import numpy as np
from flwr.common import Metrics, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log


def aggregate(results):
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


class HomomorphicFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):
        """Aggregate fit results using weighted average."""

        print("---------------------------")
        print("---------------------------")
        print("---------------------------")
        print("aggregate fit")
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        print('callers name:', [(fun[1], fun[3]) for fun in calframe])
        #print(calframe)
        print("---------------------------")
        print("---------------------------")
        print("---------------------------")

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # print("-------------------------")
        # print("-------------------------")
        # print(results)
        # print("-------------------------")
        # print("-------------------------")

        # for _, fit_res in results:
        #     print(parameters_to_ndarrays(fit_res.parameters))


        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # print("-------------------------")
        # print("-------------------------")
        # print(weights_results)
        # print("-------------------------")
        # print("-------------------------")

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated







# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define strategy
# strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)
strategy = HomomorphicFedAvg(evaluate_metrics_aggregation_fn=weighted_average)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=2),
    strategy=strategy,
)
