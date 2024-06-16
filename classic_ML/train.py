import hydra
from omegaconf import DictConfig
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

@hydra.main(config_path="config", config_name="config")
def train(cfg: DictConfig):
    # Load dataset
    if cfg.training.dataset == "iris":
        data = datasets.load_iris()
    else:
        raise ValueError("Unsupported dataset")

    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=cfg.training.test_size, random_state=cfg.training.random_state
    )

    # Initialize model
    if cfg.model._target_ == "logistic_regression":
        model = LogisticRegression(penalty=cfg.model.penalty, C=cfg.model.C, solver=cfg.model.solver)
    elif cfg.model._target_ == "random_forest":
        model = RandomForestClassifier(n_estimators=cfg.model.n_estimators, max_depth=cfg.model.max_depth, random_state=cfg.model.random_state)
    else:
        raise ValueError("Unsupported model type")

    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    train()

