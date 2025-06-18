import os

import mlflow
import torch
import munch
import toml

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

config = ""

def check_backend() -> str:
    """
    Determines the available backend engine for PyTorch computations.

    This function checks if the MPS (Metal Performance Shaders) or CUDA backends
    are available and sets the appropriate backend accordingly. If neither MPS 
    nor CUDA is available, it defaults to the CPU backend.

    Returns
    -------
    str
        The name of the backend to use for PyTorch computations ('mps', 'cuda', or 'cpu').
    """

    if torch.backends.mps.is_available():
        backend:str = 'mps'
    elif torch.cuda.is_available():
        backend:str = 'cuda'
    else:
        backend:str = 'cpu'
    
    if backend in ['mps', 'cuda']:
        matmul_precision = config.torch.matmul_precision
        torch.set_float32_matmul_precision(matmul_precision)

    return backend

def set_mlflow_endpoint(config_file):
    """
    Load info for MLFlow endpoint from toml configuration file
    """
    # read configuration file for this execution
    global config
    config = munch.munchify(toml.load(config_file))

    # define environment variables
    os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = config.mlflow.tracking_insecure_tls
    os.environ['MLFLOW_TRACKING_USERNAME'] = config.mlflow.username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = config.mlflow.password
    os.environ['MLFLOW_TRACKING_URI'] = config.mlflow.tracking_uri
    os.environ['MLFLOW_EXPERIMENT_NAME'] = config.mlflow.experiment_name

def setup_mlflow_experiment():
    """
    Configures MLflow tracking URI and sets the experiment name.
    This function should be called once at the beginning of the script.
    """
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT_NAME'))
    _log.info(f"MLflow Experiment set to '{os.getenv('MLFLOW_EXPERIMENT_NAME')}' with tracking URI '{os.getenv('MLFLOW_TRACKING_URI')}'")

def load_model_from_mlflow_registry(model_name, version=1, tag=None):
    # set tracking uri
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))

    if version:
        # Load by specific version
        model_uri = f"models:/{model_name}/{version}"
        local_path = os.path.join(os.getcwd(), 'MLFLOW', f"{model_name}/{version}")
    elif tag:
        # Load by tag (if the tag is set in the UI)
        model_uri = f"models:/{model_name}/{tag}"
        local_path = os.path.join(os.getcwd(), 'MLFLOW', f"{model_name}/{tag}")
    else:
        raise ValueError("Either version or tag must be specified for model loading.")
        
    os.makedirs(local_path, exist_ok=True)    
    model = mlflow.pytorch.load_model(model_uri, map_location=torch.device(check_backend()), dst_path=local_path)
    return model

def load_model_from_mlflow(run_name, scaler=True, provenance=False):
    # set tracking uri
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))

    run_id = mlflow.search_runs(filter_string=f"run_name='{run_name}'")['run_id'].values[0]

    local_path = os.path.join(os.getcwd(), 'MLFLOW', f"{run_name}")
    os.makedirs(local_path, exist_ok=True)
    print(f"Data from MLFlow downloaded in: {local_path}")

    client = mlflow.MlflowClient()
    if scaler:
        artifact_path = client.download_artifacts(run_id=run_id, path="scaler", dst_path=local_path)
    if provenance:
        artifact_path = client.download_artifacts(run_id=run_id, path=f"provgraph_{os.getenv('MLFLOW_EXPERIMENT_NAME')}.svg", dst_path=local_path)
        artifact_path = client.download_artifacts(run_id=run_id, path=f"provgraph_{os.getenv('MLFLOW_EXPERIMENT_NAME')}.json", dst_path=local_path)

    model_uri = f'runs:/{run_id}/last_model'
    model = mlflow.pytorch.load_model(model_uri, map_location=torch.device(check_backend()), dst_path=local_path)

    return model, local_path