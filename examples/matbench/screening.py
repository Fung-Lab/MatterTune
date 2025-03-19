from __future__ import annotations

from ase.io import read
from ase import Atoms
import numpy as np
import wandb

from mattertune.backbones import (
    EqV2BackboneModule,
    JMPBackboneModule,
    ORBBackboneModule,
)


def main(args_dict: dict):
    ## Load Checkpoint
    if "jmp" in args_dict["ckpt_path"]:
        model_type = "jmp"
        model = JMPBackboneModule.load_from_checkpoint(
            checkpoint_path=args_dict["ckpt_path"], map_location="cpu"
        )
    elif "orb" in args_dict["ckpt_path"]:
        model_type = "orb"
        model = ORBBackboneModule.load_from_checkpoint(
            checkpoint_path=args_dict["ckpt_path"], map_location="cpu"
        )
    elif "eqv2" in args_dict["ckpt_path"]:
        model_type = "eqv2"
        model = EqV2BackboneModule.load_from_checkpoint(
            checkpoint_path=args_dict["ckpt_path"], map_location="cpu", strict=False
        )
    else:
        raise ValueError(
            "Invalid fine-tuning model, must be one of 'jmp', 'orb', or 'eqv2'."
        )

    ## Load Screening Data
    atoms_list: list[Atoms] = read("/net/csefiles/coc-fung-cluster/lingyu/gnome_Bandgap.xyz", index=":") # type: ignore
    true_properties = np.array([atoms.info["bandgap"] for atoms in atoms_list])
    exclude_inf_indices = np.where(np.isinf(true_properties))[0]
    atoms_list = [atoms_list[i] for i in range(len(atoms_list)) if i not in exclude_inf_indices]
    true_properties = np.array([true_properties[i] for i in range(len(true_properties)) if i not in exclude_inf_indices])
    

    ## Run Property Prediction


    wandb.init(
        project="MatterTune-Examples",
        name="GNoME-Bandgap-Screening-{}".format(
            args_dict["ckpt_path"].split("/")[-1].split(".")[0]
        ),
        config=args_dict,
    )
    property_predictor = model.property_predictor(
        lightning_trainer_kwargs={
            "accelerator": "gpu",
            "devices": args_dict["devices"],
            "precision": "32",
            "inference_mode": False,
            "enable_progress_bar": True,
            "enable_model_summary": False,
            "logger": False,
            "barebones": False,
        }
    )
    model_outs = property_predictor.predict(
        atoms_list, batch_size=args_dict["batch_size"]
    )
    pred_properties = [out["matbench_mp_gap"].item() for out in model_outs]
    print(pred_properties)

    ## Compare Predictions
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        recall_score,
    )

    true_properties = np.array(true_properties)
    pred_properties = np.array(pred_properties)

    # Regression Metrics
    mae = mean_absolute_error(true_properties, pred_properties)
    mse = mean_squared_error(true_properties, pred_properties)
    rmse = np.sqrt(mse)

    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

    ## Screening Metrics
    thresholds = sorted(args_dict["thresholds"])
    lower_bound, upper_bound = thresholds[0], thresholds[1]

    true_labels = (
        (true_properties >= lower_bound) & (true_properties < upper_bound)
    ).astype(int)
    pred_labels = (
        (pred_properties >= lower_bound) & (pred_properties < upper_bound)
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()

    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")

    accuracy = accuracy_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1 Score: {f1:.4f}")

    wandb.log(
        {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "Accuracy": accuracy,
            "Recall": recall,
            "F1 Score": f1,
        }
    )

    ## Plot Bandgap Distribution
    sorted_indices = np.argsort(true_properties)
    true_properties = true_properties[sorted_indices]
    pred_properties = pred_properties[sorted_indices]
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 3))
    plt.plot(pred_properties, label="Predicted Bandgap", alpha=0.5)
    plt.plot(true_properties, label="True Bandgap", alpha=0.5)
    plt.xlabel("Index")
    plt.ylabel("Bandgap (eV)")
    plt.legend()
    # plt.yscale("log")

    wandb.log({"Bandgap Distribution": plt})

    plt.savefig(f"./plots/{model_type}-gnome.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path", type=str, default="./checkpoints-matbench_mp_gap/orb-best-fold0.ckpt"
    )
    parser.add_argument("--devices", type=int, nargs="+", default=[2])
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[1.0, 3.0])
    args = parser.parse_args()

    main(vars(args))
