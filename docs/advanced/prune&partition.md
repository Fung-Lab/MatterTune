# Accelerate Foundation Models via Message-Passing Pruning and Graph Partitioning

## Environment Setup

If you do not need to use this foundation model acceleration method, then the environment configuration following the [installation guidance](https://fung-lab.github.io/MatterTune/installation.html) is entirely sufficient. However, if you wish to use this acceleration method based on message-passing pruning and graph partitioning, some minor code modifications to the backbone model codebase are required. These modifications are minimal, and for user convenience, we directly provide the modified backbone model codebase. Users only need to follow the installation guidance while replacing the backbone model source with our modified repository listed below.
- [MatterSim](https://github.com/Lingyu-Kong/MatterSim-MT)
- [MACE](https://github.com/Lingyu-Kong/MACE-MT)
- [ORB](https://github.com/Lingyu-Kong/ORB-MT)
- [JMP](https://github.com/Lingyu-Kong/JMP-MT)

Let us explain the modifications we made. 
- First, we need the forward function of the backbone model to allow the loop over the message-passing layers to be truncated according to the specified maximum number of message-passing steps. For example, in the MatterSim model, the modification we made is as follows:

    ```python
    # src/mattersim/forcefield/m3gnet/m3gnet.py
    # In forward function, line 134
            # New Main Loop
            for i in range(self.num_blocks):
                atom_attr, edge_attr = self.graph_conv[i](
                    atom_attr,
                    edge_attr,
                    edge_attr_zero,
                    edge_index,
                    three_basis,
                    three_body_indices,
                    edge_length,
                    num_bonds,
                    num_triple_ij,
                    num_atoms,
                )
                if return_intermediate:
                    internal_attrs[f"node_attr_{i}"] = atom_attr.clone()
                    internal_attrs[f"edge_attr_{i}"] = edge_attr.clone()

            # Old Main Loop
            # for idx, conv in enumerate(self.graph_conv):
            #     atom_attr, edge_attr = conv(
            #         atom_attr,
            #         edge_attr,
            #         edge_attr_zero,
            #         edge_index,
            #         three_basis,
            #         three_body_indices,
            #         edge_length,
            #         num_bonds,
            #         num_triple_ij,
            #         num_atoms,
            #     )
            #     if return_intermediate:
            #         internal_attrs[f"node_attr_{idx}"] = atom_attr.clone()
            #         internal_attrs[f"edge_attr_{idx}"] = edge_attr.clone()
    ```
- Another important modification is enabling the model to return per-node energy predictions alongside the total energy prediction. This is necessary to aggregate the predictions from each subgraph during graph-partition-based parallelization.  


## Model Fine-tuning with Message-Passing Pruning

To apply message-passing pruning during model fine-tuning, it is only necessary to set the ```model.pruning_message_passing``` parameter of the ```MC.MatterTunerConfig``` object to the desired number of message-passing layers to retain. For details on usage, please refer to line 63 of [Li3PO4/train.py](https://github.com/Fung-Lab/MatterTune/blob/main/examples/prune_and_partition/Li3PO4/train.py)

## Multi-GPU Parallelism Based on Graph Partitioning

We provide the ```MatterTunePartitionCalculator```, a class that inherits from the ASE Calculator and can automatically handle graph partitioning and multi-GPU parallelism. It should be noted that when constructing an instance of this class, a ```ParallizedInferenceDDP``` instance must be provided as the executor for multi-GPU parallel inference. For details on usage, please refer to line 59 to line 73 of [Li3PO4/md.py](https://github.com/Fung-Lab/MatterTune/blob/main/examples/prune_and_partition/Li3PO4/md.py)

## Examples

In the [MatterTune/examples/prune_and_partition](https://github.com/Fung-Lab/MatterTune/tree/main/examples/prune_and_partition) directory, we provide three examples, including fine-tuning a model with message-passing pruning applied, as well as using the fine-tuned model for MD simulations.
