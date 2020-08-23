from packaging import version
import tensorflow as tf
import optuna
from optuna.samplers import TPESampler

from train import YoloTrain

pretrained_ckpt = "/home/jsteeg/tensorflow-yolov3/training2020/08-09y1/best/yolov3_coco_ufo_S1_test_loss=4.7399.ckpt-9"
EPOCHS = 2

if version.parse(tf.__version__) < version.parse("1.14.0"):
    raise RuntimeError("tensorflow>=1.14.0 is required for this example.")

def create_model(trial, batch_size, optimizer):
    return YoloTrain(2, hyperparameter_search = True, batch_size = batch_size, optimizer = optimizer)

def create_optimizer(trial):
    kwargs = {}
    optimizer_options = ['AdamOptimizer', 'MomentumOptimizer']
    optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
    if optimizer_selected == "AdamOptimizer":
        kwargs["learning_rate"] = trial.suggest_uniform("adam_learning_rate", 1e-7, 1e-3)
        kwargs["beta1"] = trial.suggest_uniform("beta1", 0.9, 0.999)
        kwargs["beta2"] = trial.suggest_uniform("beta2", 0.99, 0.9999)
        kwargs["epsilon"] = trial.suggest_uniform("epsilon", 1e-8, 1e-9,)
    elif optimizer_selected == "MomentumOptimizer":
        kwargs["learning_rate"] = trial.suggest_float(
            "sgd_opt_learning_rate", 1e-7, 1e-3, log = True
        )
        kwargs["momentum"] = trial.suggest_float("sgd_opt_momentum", 1e-5, 1e-1, log =True)
    else:
        assert False, "ERROR: Got {} as optimizer".format(optimizer_selected)

    optimizer = getattr(tf.compat.v1.train, optimizer_selected)(**kwargs)

    return optimizer


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):

    # Build model and optimizer.
    
    optimizer = create_optimizer(trial)
    batch_size = trial.suggest_categorical("batch_size", [4, 6, 8])

    
    model = create_model(trial, batch_size, optimizer)
    
    isPruned = False
    with tf.device("/cpu:0"):
        model.initialize_session(pretrained_ckpt)

        for epoch in range(EPOCHS):
            print("Training...")
            model.optimize_hyperparameters(dataset = "train", isTrainable = True)

            print("Testing...")
            test_loss = model.optimize_hyperparameters(dataset = "test", isTrainable = False)

            trial.report(test_loss, epoch)

            if trial.should_prune():
                isPruned = True
                break
                
    # TODO Replace with solution to start model without re-initializing the whole graph
    tf.compat.v1.reset_default_graph()

    if isPruned:
        raise optuna.TrialPruned()

    return test_loss


if __name__ == "__main__":
    # 'create_study(...)' to use only one script
    # TODO: for distribution over multiple machines, create a master/slave script
    study = optuna.create_study(
        direction = "minimize",
        study_name = "yolov3_study",
        sampler = TPESampler(),
        pruner = optuna.pruners.MedianPruner(),
        storage = "sqlite:///yolov3_study.db",
        load_if_exists = True )
    study.optimize(objective, n_trials = 2)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


    results = study.trials_dataframe()
    results.to_csv("./yolo_study_results.csv")
