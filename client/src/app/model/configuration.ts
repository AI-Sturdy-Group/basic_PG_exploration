export interface Configuration {
    name: string;
    desc: string;
    training_steps: number;
    show_every: number;
    learning_rate: number;
    experience_size: number;
    minibatch_size: number;
    hidden_layer_sizes: number[];
    hidden_activation: string;
    actions_size: number;
    save_policy_every: number;
    mu_activation: string;
    sigma_activation: string;
    true_action: number;
    start_mu: number;
    start_sigma: number;
    replace: boolean;
    normalize_rewards: boolean;
}
