from __future__ import annotations

import logging
from pathlib import Path
import os
import json

import h5py
import numpy as np
import torch
from ase import Atoms
from ase.io import read, write
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
from matplotlib.colors import ListedColormap
from pymatgen.io.ase import AseAtomsAdaptor
from matbench.bench import (  # type: ignore[reportMissingImports] # noqa
    MatbenchBenchmark,
)

import mattertune.configs as MC
from mattertune.finetune.base import FinetuneModuleBase, ModelOutput
from mattertune.wrappers.property_predictor import MatterTunePropertyPredictor
from mattertune.wrappers.property_predictor import _atoms_list_to_dataloader
from mattertune.backbones import (
    MatterSimM3GNetBackboneModule,
    JMPBackboneModule,
    ORBBackboneModule,
    EqV2BackboneModule,
)

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

def save_internal_features_h5(internal_features, filename="internal_features.h5"):
    with h5py.File(filename, "w") as f:
        g = f.create_group("features")
        for i, feature_dict in enumerate(internal_features):
            subgroup = g.create_group(str(i))
            for k, arr in feature_dict.items():
                if np.isscalar(arr):
                    arr = np.array([arr])
                if issubclass(arr.dtype.type, np.floating):
                    arr = arr.astype(np.float16)
                subgroup.create_dataset(
                    k,
                    data=arr,
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True
                )
    print(f"Features saved to {filename}")

def load_internal_features_h5(filename="internal_features.h5"):
    internal_features = []
    with h5py.File(filename, "r") as f:
        g = f["features"]
        for i in sorted(g.keys(), key=lambda x: int(x)):
            subgroup = g[i]
            feature_dict = {}
            for k in subgroup.keys():
                feature_dict[k] = np.array(subgroup[k])
            internal_features.append(feature_dict)
    return internal_features

def get_periodic_table_color():
    periodic_table_color: dict = json.load(open("periodic_table_color.json"))
    
    color_array = []
    for i in range(1, 119):
        if str(i) in periodic_table_color:
            color = periodic_table_color[str(i)]
            color_array.append(color)
        else:
            # assign white color to unknown elements
            color_array.append("#ffffff")
    
    periodic_table_cmap = ListedColormap(color_array, name="periodic_table")
    return periodic_table_cmap

def load_pretrained_model(model_type: str):
    if "mattersim" in model_type:
        model_config = MC.MatterSimBackboneConfig.draft()
        model_config.graph_convertor = MC.MatterSimGraphConvertorConfig.draft()
        match model_type.lower():
            case "mattersim-1m":
                model_config.pretrained_model = "MatterSim-v1.0.0-1M"
            case "mattersim-5m":
                model_config.pretrained_model = "MatterSim-v1.0.0-5M"
            case _:
                raise ValueError(f"Unknown model type: {model_type}")
        model_config.use_pretrained_normalizers = False
    elif "orb" in model_type:
        model_config = MC.ORBBackboneConfig.draft()
        model_config.pretrained_model = model_type.lower()
        model_config.use_pretrained_normalizers = False
    elif "jmp" in model_type:
        model_config = MC.JMPBackboneConfig.draft()
        model_config.graph_computer = MC.JMPGraphComputerConfig.draft()
        model_config.graph_computer.pbc = True
        model_config.pretrained_model = model_type.lower()
        model_config.use_pretrained_normalizers = False
    elif "eqv2" in model_type:
        model_config = MC.EqV2BackboneConfig.draft()
        model_config.checkpoint_path = Path(
            "/net/csefiles/coc-fung-cluster/nima/shared/checkpoints/eqV2_31M_mp.pt"
        )
        model_config.atoms_to_graph = MC.FAIRChemAtomsToGraphSystemConfig.draft()
        model_config.atoms_to_graph.radius = 8.0
        model_config.atoms_to_graph.max_num_neighbors = 20
        model_config.use_pretrained_normalizers = False
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model_config.freeze_backbone = True
    model_config.optimizer = MC.AdamWConfig(lr=8.0e-5)
    model_config.properties = []
    energy = MC.EnergyPropertyConfig(
        loss=MC.MSELossConfig(),
        loss_coefficient=1.0,
    )
    model_config.properties.append(energy)
    forces = MC.ForcesPropertyConfig(
        loss=MC.MSELossConfig(),
        loss_coefficient=1.0,
        conservative=True if "mattersim" in model_type else False,
    )
    model_config.properties.append(forces)

    model = model_config.create_model()
    return model

def load_finetuned_model(model_type: str, checkpoint_path: str):
    if "jmp" in model_type:
        model = JMPBackboneModule.load_from_checkpoint(checkpoint_path, strict=False)
    elif "orb" in model_type:
        model = ORBBackboneModule.load_from_checkpoint(checkpoint_path, strict=False)
    elif "mattersim" in model_type:
        model = MatterSimM3GNetBackboneModule.load_from_checkpoint(checkpoint_path, strict=False)
    elif "eqv2" in model_type:
        model = EqV2BackboneModule.load_from_checkpoint(checkpoint_path, strict=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model

def get_atoms_list(dataset: str, num_samples: int | None = None):
    if dataset == "mptraj":
        atoms_list: list[Atoms] = read(os.path.join("/net/csefiles/coc-fung-cluster/lingyu/datasets", "mptraj_train.xyz"), ":") # type: ignore
        atoms_list.extend(read(os.path.join("/net/csefiles/coc-fung-cluster/lingyu/datasets", "mptraj_val.xyz"), ":")) # type: ignore
        atoms_list.extend(read(os.path.join("/net/csefiles/coc-fung-cluster/lingyu/datasets", "mptraj_test.xyz"), ":")) # type: ignore
    elif dataset == "wbm":
        atoms_list = read(os.path.join("/net/csefiles/coc-fung-cluster/lingyu/datasets", "wbm.xyz"), ":") # type: ignore
    elif "matbench" in dataset:
        mb = MatbenchBenchmark(autoload=False, subset=[dataset])
        task = list(mb.tasks)[0]
        task.load()
        
        def data_convert(structures, properties=None):
            adapter = AseAtomsAdaptor()
            atoms_list = []
            for i, structure in enumerate(structures):
                atoms = adapter.get_atoms(structure)
                assert isinstance(atoms, Atoms), "Expected an Atoms object"
                if properties is not None:
                    atoms.info[dataset] = float(properties[i])
                atoms_list.append(atoms)
            return atoms_list

        fold_i = task.folds[0]
        inputs_data, outputs_data = task.get_train_and_val_data(fold_i)
        atoms_list_train = data_convert(inputs_data, outputs_data)
        for atoms in atoms_list_train:
            atoms.info["split"] = 0
        inputs_data, outputs_data = task.get_test_data(fold_i, include_target=True)
        atoms_list_test = data_convert(inputs_data, outputs_data)
        for atoms in atoms_list_test:
            atoms.info["split"] = 1
        atoms_list = atoms_list_train + atoms_list_test
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    if num_samples is not None:
        sample_indices = np.random.choice(len(atoms_list), min(num_samples, len(atoms_list)), replace=False)
        atoms_list = [atoms_list[i] for i in sample_indices]
    return atoms_list

def evaluate_single_point(dataset: str, predictor: MatterTunePropertyPredictor, atoms: Atoms):
    if dataset == "mptraj" or dataset == "wbm":
        e_ori, f_ori = atoms.get_potential_energy(), np.array(atoms.get_forces())
        pred = predictor.predict([atoms])[0]
        e_pred, f_pred = pred["energy"].item(), pred["forces"].detach().cpu().numpy()
        f_aes = np.linalg.norm(f_ori - f_pred, axis=1).tolist()
        return f_aes
    elif "matbench" in dataset:
        p_ori = atoms.info[dataset]
        pred = predictor.predict([atoms])[0]
        p_pred = pred[dataset].item()
        ae = [abs(p_ori - p_pred)] * len(atoms)
        return ae
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def main(args_dict: dict):
    internal_features = []
    if not args_dict["load"]:
        if args_dict["pt_or_ft"] == "pt":
            model = load_pretrained_model(args_dict["model_type"])
        else:
            model = load_finetuned_model(args_dict["model_type"], args_dict["checkpoint_path"])
        
        internal_feature_preditor = model.internal_feature_predictor(
            lightning_trainer_kwargs={
                "accelerator": "gpu",
                "devices": [args_dict["cuda_devices"][0]],
                "precision": "32",
                "inference_mode": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "logger": False,
                "barebones": True,
            }
        )
        property_predictor = model.property_predictor(
            lightning_trainer_kwargs={
                "accelerator": "gpu",
                "devices": [args_dict["cuda_devices"][1]],
                "precision": "32",
                "inference_mode": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "logger": False,
                "barebones": True,
            }
        )
        atoms_list = get_atoms_list(args_dict["dataset"], args_dict["num_samples"])
        
        for idx, atoms in enumerate(tqdm(atoms_list)):
            backbone_output = internal_feature_preditor.predict([atoms])[0]
            internal_feature = {"idx": idx, "atomic_numbers": np.array(atoms.get_atomic_numbers())}
            if "split" in atoms.info:
                internal_feature["split"] = atoms.info["split"]
            try:
                errors = evaluate_single_point(args_dict["dataset"], property_predictor, atoms)
            except Exception as e:
                # logging.error(f"Error occurred when evaluating atom {idx}: {e}")
                errors = [0.0] * len(atoms)
            internal_feature["node_errors"] = np.array(errors)
            for key, value in backbone_output.items():
                if type(value) == list:
                    internal_feature[key] = np.array([v.detach().cpu().numpy() for v in value])
                else:
                    try:
                        internal_feature[key] = value.detach().cpu().numpy()
                    except Exception as e:
                        internal_feature[key] = np.array(value)
            internal_features.append(internal_feature)
    #     save_internal_features_h5(
    #         internal_features,
    #         filename=f"/storage/lingyu/internal_features/{args_dict['dataset']}_{args_dict['model_type']}_{args_dict['pt_or_ft']}_internal_features.h5",
    #     )
        
    # internal_features = load_internal_features_h5(
    #     filename=f"/storage/lingyu/internal_features/{args_dict['dataset']}_{args_dict['model_type']}_{args_dict['pt_or_ft']}_internal_features.h5"
    # )
    keys = internal_features[0].keys()
    periodic_table_cmap = get_periodic_table_color()
    for key in keys:
        if key == "idx" or key == "atomic_numbers" or key == "node_errors" or key == "split":
            continue
        atomic_numbers_0 = internal_features[0]["atomic_numbers"]
        value_0 = internal_features[0][key]
        if len(value_0) == len(atomic_numbers_0):
            embedding_list = []
            atomic_numbers_list = []
            error_list = []
            split_list = []
            for idx in range(len(internal_features)):
                atomic_numbers = internal_features[idx]["atomic_numbers"]
                error = internal_features[idx]["node_errors"]
                embedding = internal_features[idx][key]
                for j_idx in range(len(atomic_numbers)):
                    atomic_numbers_list.append(atomic_numbers[j_idx])
                    embedding_list.append(embedding[j_idx])
                    error_list.append(error[j_idx])
                if "split" in keys:
                    split = internal_features[idx]["split"]
                    split_list.extend([split] * len(atomic_numbers))
                else:
                    split_list.extend([0] * len(atomic_numbers))
            embedding_list = np.array(embedding_list)
            if "eqv2" in args_dict["model_type"]:
                embedding_list = embedding_list[:, 0, :]
                embedding_list = embedding_list.reshape(embedding_list.shape[0], -1)
            atomic_numbers_list = np.array(atomic_numbers_list)
            error_list = np.array(error_list)

            fig, axs = plt.subplots(2, 3, figsize=(18, 12))
            # PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(embedding_list)
            train_indices = np.where(np.array(split_list) == 0)[0]
            train_pca_result = pca_result[train_indices]
            test_indices = np.where(np.array(split_list) == 1)[0]
            test_pca_result = pca_result[test_indices]
            # dot markers for train and square markers for test
            axs[0, 0].scatter(train_pca_result[:, 0], train_pca_result[:, 1], c=error_list[train_indices], cmap="coolwarm", s=10, marker="o", alpha=0.8)
            axs[0, 0].scatter(test_pca_result[:, 0], test_pca_result[:, 1], c=error_list[test_indices], cmap="coolwarm", s=20, marker="x")
            axs[0, 0].set_title(f"{key}-node_error PCA")
            axs[1, 0].scatter(train_pca_result[:, 0], train_pca_result[:, 1], c=atomic_numbers_list[train_indices], cmap=periodic_table_cmap, s=10, marker="o")
            axs[1, 0].scatter(test_pca_result[:, 0], test_pca_result[:, 1], c=atomic_numbers_list[test_indices], cmap=periodic_table_cmap, s=20, marker="x")
            axs[1, 0].set_title(f"{key}-atomic_numbers PCA")
            
            # TSNE
            tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
            tsne_result = tsne.fit_transform(embedding_list)
            train_indices = np.where(np.array(split_list) == 0)[0]
            train_tsne_result = tsne_result[train_indices]
            test_indices = np.where(np.array(split_list) == 1)[0]
            test_tsne_result = tsne_result[test_indices]
            axs[0, 1].scatter(train_tsne_result[:, 0], train_tsne_result[:, 1], c=error_list[train_indices], cmap="coolwarm", s=10, marker="o", alpha=0.8)
            axs[0, 1].scatter(test_tsne_result[:, 0], test_tsne_result[:, 1], c=error_list[test_indices], cmap="coolwarm", s=20, marker="x")
            axs[0, 1].set_title(f"{key}-node_error TSNE")
            axs[1, 1].scatter(train_tsne_result[:, 0], train_tsne_result[:, 1], c=atomic_numbers_list[train_indices], cmap=periodic_table_cmap, s=10, marker="o")
            axs[1, 1].scatter(test_tsne_result[:, 0], test_tsne_result[:, 1], c=atomic_numbers_list[test_indices], cmap=periodic_table_cmap, s=20, marker="x")
            axs[1, 1].set_title(f"{key}-atomic_numbers TSNE")
            
            # UMAP
            umap = UMAP(n_components=2, n_neighbors=30, min_dist=0.0)
            umap_result = np.array(umap.fit_transform(embedding_list))
            train_indices = np.where(np.array(split_list) == 0)[0]
            train_umap_result = umap_result[train_indices]
            test_indices = np.where(np.array(split_list) == 1)[0]
            test_umap_result = umap_result[test_indices]
            axs[0, 2].scatter(train_umap_result[:, 0], train_umap_result[:, 1], c=error_list[train_indices], cmap="coolwarm", s=10, marker="o", alpha=0.8)
            axs[0, 2].scatter(test_umap_result[:, 0], test_umap_result[:, 1], c=error_list[test_indices], cmap="coolwarm", s=20, marker="x")
            axs[0, 2].set_title(f"{key}-node_error UMAP")
            sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=min(error_list), vmax=max(error_list)))
            cbar = plt.colorbar(sm, ax=axs[0, 2])
            cbar.set_label("Node Error")
            axs[1, 2].scatter(train_umap_result[:, 0], train_umap_result[:, 1], c=atomic_numbers_list[train_indices], cmap=periodic_table_cmap, s=10, marker="o")
            axs[1, 2].scatter(test_umap_result[:, 0], test_umap_result[:, 1], c=atomic_numbers_list[test_indices], cmap=periodic_table_cmap, s=20, marker="x")
            axs[1, 2].set_title(f"{key}-atomic_numbers UMAP")
            sm = plt.cm.ScalarMappable(cmap=periodic_table_cmap, norm=plt.Normalize(vmin=1, vmax=118))
            cbar = plt.colorbar(sm, ax=axs[1, 2])
            cbar.set_label("Atomic Number")

            if not os.path.exists(f"./results/{args_dict['model_type']}_{args_dict['pt_or_ft']}"):
                os.makedirs(f"./results/{args_dict['model_type']}_{args_dict['pt_or_ft']}")
            plt.savefig(f"./results/{args_dict['model_type']}_{args_dict['pt_or_ft']}/{args_dict['dataset']}-{key}.png")
            
            

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--pt_or_ft", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--cuda_devices", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()
    
    assert args.pt_or_ft in ["pt", "ft"]
    if args.pt_or_ft == "ft" and args.checkpoint_path is None:
        raise ValueError("Please provide the checkpoint path for fine-tuned model")
    
    if args.wandb:
        wandb.login()
        wandb.init(
            project="MatterTune-Internal-Feature-Export",
            name=f"{args.model_type}-{args.pt_or_ft}-{args.dataset}",
        )
    
    args_dict = vars(args)
    main(args_dict)