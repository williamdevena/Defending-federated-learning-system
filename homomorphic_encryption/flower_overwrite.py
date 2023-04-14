import inspect
import pickle
from dataclasses import dataclass
from io import BytesIO
from typing import List, cast

import numpy as np
from flwr.common.typing import Code, NDArray, NDArrays, Parameters, Status


@dataclass
class ParametersNew:
    """Model parameters."""

    tensors: List
    tensor_type: str



@dataclass
class GetParametersResNew:
    """Response when asked to return parameters."""

    status: Status
    parameters: ParametersNew




def get_parameters2_new(self, ins):
    """Return the current local model parameters."""
    print("---------------------------")
    print("---------------------------")
    print("---------------------------")
    print("---------------------------")
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)
    print('callers name:', [(fun[1], fun[3]) for fun in calframe])
    #print(calframe)
    print("---------------------------")
    print("---------------------------")
    print("---------------------------")
    print("_get_parameters1")
    parameters = self.numpy_client.get_parameters(config=ins.config)  # type: ignore
    print("_get_parameters2")
    parameters_proto = ndarrays_to_parameters_new(parameters)
    print("_get_parameters3")
    return GetParametersResNew(
        status=Status(code=Code.OK, message="Success"), parameters=parameters_proto
    )



def parameters_to_proto_new(parameters):
    """Serialize `Parameters` to ProtoBuf."""
    return ParametersNew(tensors=parameters.tensors, tensor_type=parameters.tensor_type)



def ndarrays_to_parameters_new(ndarrays):
    """Convert NumPy ndarrays to parameters object."""
    print("------------------------- BELAAAAAAAAA ---------------")
    print("------------------------- BELAAAAAAAAA ---------------")
    print("------------------------- BELAAAAAAAAA ---------------")
    print("------------------------- BELAAAAAAAAA ---------------")
    print("ndarrays_to_param")
    #print(ndarrays)
    #tensors = [ndarray_to_bytes_new(ndarray) for ndarray in ndarrays]
    tensors = [ndarray for ndarray in ndarrays]
    #print("----------------------------")
    #print(tensors)

    return ParametersNew(tensors=tensors, tensor_type="numpy.ndarray")



def parameters_to_ndarrays_new(parameters: Parameters) -> NDArrays:
    """Convert parameters object to NumPy ndarrays."""
    print("------------------------- BELAAAAAAAAA ---------------")
    print("------------------------- BELAAAAAAAAA ---------------")
    print("------------------------- BELAAAAAAAAA ---------------")
    print("------------------------- BELAAAAAAAAA ---------------")
    print("parameters_to_ndarrays")
    return [bytes_to_ndarray_new(tensor) for tensor in parameters.tensors]
    #return [tensor for tensor in parameters.tensors]



def ndarray_to_bytes_new(ndarray):
    """Serialize NumPy ndarray to bytes."""
    print("ndarray_to_bytes")
    #bytes_io = BytesIO()
    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html

    #np.save(bytes_io, ndarray, allow_pickle=False)  # type: ignore
    # print("------------------------- BELAAAAAAAAA ---------------")
    # print("------------------------- BELAAAAAAAAA ---------------")
    # print("------------------------- BELAAAAAAAAA ---------------")
    # print("------------------------- BELAAAAAAAAA ---------------")
    #dec_ndarray = ndarray.decrypt()
    #print(ndarray, dec_ndarray)
    #print(ndarray)

    #np.save(bytes_io, dec_ndarray, allow_pickle=True)  # type: ignore
    #np.save(bytes_io, ndarray, allow_pickle=True)  # type: ignore

    #bytes_io = BytesIO(ndarray)
    #bytes_io = pickle.dumps(ndarray)
    bytes_io = bytes(ndarray)

    #print(bytes_io.getvalue())
    return bytes_io.getvalue()



def bytes_to_ndarray_new(tensor):
    """Deserialize NumPy ndarray from bytes."""
    bytes_io = BytesIO(tensor)
    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.load.html
    ndarray_deserialized = np.load(bytes_io, allow_pickle=True)  # type: ignore
    return cast(NDArray, ndarray_deserialized)