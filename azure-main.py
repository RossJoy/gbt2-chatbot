# get-started/run-hello.py
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig

ws = Workspace.from_config()

experiment = Experiment(workspace=ws, name='thesis-deneme')

myenv = Environment.from_pip_requirements(name = "myenv",
                                          file_path = "./requirements.txt")

config = ScriptRunConfig(source_directory='./src', script='main.py', compute_target='cpudeneme', environment=myenv)

run = experiment.submit(config)
aml_url = run.get_portal_url()
print(aml_url)