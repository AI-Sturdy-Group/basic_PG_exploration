import logging
import json
import time
import shutil
from pathlib import Path

import git
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from agents import NaivePolicyGradientAgent, BaseAgentConfig
from environments import SimpleContinuousEnvironment
from models import SimpleModel
from code_utils import prepare_stream_logger, prepare_file_logger


logger = logging.getLogger()
prepare_stream_logger(logger, logging.INFO)
logging.getLogger("tensorflow").setLevel(logging.ERROR)


app = Flask(__name__)
cors = CORS(app)


DEFAULT_NAIVE_CONFIG = {
    "name": "test_run",
    "desc": "Test experiment.",
    "training_steps": 100,
    "show_every": 5,
    "learning_rate": 0.001,
    "experience_size": 5,
    "minibatch_size": None,
    "hidden_layer_sizes": [],
    "hidden_activation": "tanh",
    "actions_size": 1,
    "save_policy_every": 5,
    "mu_activation": "tanh",
    "sigma_activation": "softplus"
}

# Home page
@app.route("/")
def home():

    return render_template("index.html", arguments=DEFAULT_NAIVE_CONFIG)


@app.route("/start_training", methods=["POST", "GET"])
def start_training():

    if request.method == "POST":
        args_dict = request.get_json()
        print(args_dict)

        agent_type = "naive"  # TODO: Make variable
        agent_path = Path("experiments", agent_type, args_dict["name"])
        agent_config = BaseAgentConfig(config_dict=args_dict)

        # # Get git version
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha

        # Create experiment folder and handle old results
        if agent_path.exists():
            shutil.rmtree(agent_path)
        agent_path.mkdir(parents=True)

        # Save experiments configurations and start experiment log
        prepare_file_logger(logger, logging.INFO, Path(agent_path, "experiment.log"))
        logger.info(f"Running {agent_type} policy gradient on SimpleContinuous")
        agent_config.log_configurations(logger)
        experiment_config_file = Path(agent_path, "configurations.json")
        logger.info(f"Saving experiment configurations to {experiment_config_file}")
        agent_config.to_json_file(experiment_config_file)

        env = SimpleContinuousEnvironment()
        policy = SimpleModel(model_path=Path(agent_path, "model"),
                            layer_sizes=agent_config.hidden_layer_sizes,
                            learning_rate=agent_config.learning_rate,
                            actions_size=agent_config.actions_size,
                            hidden_activation=agent_config.hidden_activation,
                            mu_activation=agent_config.mu_activation,
                            sigma_activation=agent_config.sigma_activation)
        agent = NaivePolicyGradientAgent(env=env,
                                        agent_path=agent_path,
                                        policy=policy,
                                        agent_config=agent_config)

        start_time = time.time()
        test_reward = agent.train_policy(train_steps=agent_config.training_steps,
                                        experience_size=agent_config.experience_size,
                                        show_every=agent_config.show_every,
                                        save_policy_every=agent_config.save_policy_every,
                                        minibatch_size=agent_config.minibatch_size)
        train_time = time.time() - start_time

        experiment_info = {"mean_test_reward": float(test_reward),
                        "description": agent_config.desc,
                        "git_hash": sha,
                        "train_time": train_time}

        with open(Path(agent_path, "experiment_information.json"), "w") as outfile:
            json.dump(experiment_info, outfile, indent=4)

        return experiment_info, 200




if __name__ == "__main__":
    app.run()
