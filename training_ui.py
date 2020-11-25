from flask import Flask, render_template, request

app = Flask(__name__)


ARGUMENTS = ["name", "desc", "training_steps", "show_every", "learning_rate",
             "experience_size", "minibatch_size", "hidden_layer_sizes",
             "activation", "save_policy_every", "actions_size", "mu_activation",
             "sigma_activation"]


# Home page
@app.route("/")
def home():

    return render_template("index.html", arguments=ARGUMENTS)


@app.route("/start_training", methods=["POST"])
def start_training():

    args_dict = {}
    for name in ARGUMENTS:
        args_dict[name] = request.form[name]
    return args_dict


if __name__ == "__main__":
    app.run()
