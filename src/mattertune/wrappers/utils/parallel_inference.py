from __future__ import annotations

import os
import time
import copy
import tempfile
import subprocess
import shutil
import weakref

from abc import ABC, abstractmethod
from ase import Atoms
from ase.io import read, write
import numpy as np
import torch
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing_extensions import override, cast

from mattertune.finetune.base import FinetuneModuleBase
from mattertune.util import load_from_npz, write_to_npz
from mattertune.wrappers.property_predictor import _atoms_list_to_dataloader, _create_trainer
from mattertune.wrappers.utils.multi_gpu_ddp import (
    JOB_NEW_INPUT_SIGNAL,
    JOB_FINISH_SIGNAL,
    JOB_FAILURE_SIGNAL,
    EXIT_SIGNAL,
    job_new_input_signal,
    exit_signal,
)


def is_file_stable(file_path: str, delay: float = 0.02) -> bool:
    """Check if the file size is stable for a certain period of time."""
    if not os.path.exists(file_path):
        return False
    initial_size = os.path.getsize(file_path)
    time.sleep(delay)
    return os.path.getsize(file_path) == initial_size


class SingleGPUInferenceHandler(FileSystemEventHandler):
    """
    We implement a custom event handler that will be triggered when the file is created.
    This Handler spys on the specified file path and run the inference code when the file is created.
    After the inference is done, it will delete the file.
    This is useful for multi-gpu inference in serial.
    """
    def __init__(
        self,
        *,
        input_file_path: str,
        output_file_path: str,
        model: FinetuneModuleBase,
        device: str,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.model = copy.deepcopy(model).to(self.device) # create a new model copy and move it to the device
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        
        self.build_graph_time_list = []
        self.forward_time_list = []
        
    def _run_inference(
        self,
    ):  
        trials = 0
        atoms_list = None # type: ignore
        while trials < 100:
            try:
                if self.input_file_path.endswith(".npz"):
                    atoms_list: list[Atoms] = load_from_npz(self.input_file_path)
                elif self.input_file_path.endswith(".xyz"):
                    atoms_list: list[Atoms] = read(self.input_file_path, ":") # type: ignore
                else:
                    raise ValueError(f"Unsupported file type: {self.input_file_path}")
                break
            except Exception as e:
                trials += 1
                time.sleep(0.1)
                continue
        assert atoms_list is not None, f"Failed to load the input file: {self.input_file_path}, try {trials} times"
        time1 = time.time()
        dataloader = _atoms_list_to_dataloader(
            atoms_list, self.model, batch_size=self.batch_size, num_workers=min(self.num_workers, len(atoms_list))
        )
        self.build_graph_time_list.append(time.time() - time1)
        time1 = time.time()
        predictions = []
        for batch in dataloader:
            batch = batch.to(self.device)
            pred = self.model.predict_step(
                batch = batch,
                batch_idx = 0,
            )
            predictions.extend(pred) # type: ignore
        self.forward_time_list.append((time.time() - time1)/len(atoms_list))
        torch.save(predictions, self.output_file_path)
        
    def process(self, event_path: str | bytes):
        """
        Process the event when the file is created.
        """
        if isinstance(event_path, bytes):
            event_path = os.fsdecode(event_path)
        if event_path == self.input_file_path:
            while not is_file_stable(self.input_file_path):
                time.sleep(0.01)
            time.sleep(0.05)
            self._run_inference()
            os.remove(self.input_file_path)
        else:
            pass
    
    @override
    def on_created(self, event):
        self.process(event.src_path) # type: ignore

    @override
    def on_modified(self, event):
        # self.process(event.src_path) # type: ignore
        pass
    
    
class ParallizedInferenceBase(ABC):
    name: str
    @abstractmethod
    def run_inference(
        self,
        atoms_list: list[Atoms],
    ) -> list[dict[str, torch.Tensor]]:
        weakref.finalize(self, ParallizedInferenceBase.exit, self)
    
    @abstractmethod
    def exit(self): ...
        
        
class ParallizedInferenceMultiWatcher(ParallizedInferenceBase):
    """
    This class is used to run the inference in parallel.
    """
    name = "ParallizedInferenceMultiWatcher"
    
    def __init__(
        self,
        *,
        tmp_dir: str | None = None,
        model: FinetuneModuleBase,
        devices: list[int],
        batch_size: int,
        num_workers: int,
        using_partition: bool,
        properties: list[str] | None = None,
        struct_format: str = "npz",
    ):
        super().__init__()
        if tmp_dir is not None and not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir, exist_ok=True)
        self.tmp_dir = tempfile.mkdtemp(dir=tmp_dir)
        os.system(f"rm -rf {self.tmp_dir}/*")
        assert struct_format in ["npz", "xyz"], f"Unsupported file type: {struct_format}"
        self.struct_format = struct_format
        
        model.hparams.using_partition = using_partition
        implemented_properties: list[str] = []
        for prop in model.hparams.properties:
            if (ase_prop_name := prop.ase_calculator_property_name()) is None:
                continue
            implemented_properties.append(ase_prop_name)
        if properties is None:
            properties = implemented_properties
        diabled_properties = list(set(implemented_properties) - set(properties))
        model.set_disabled_heads(diabled_properties)
        self.model = model
        
        self.devices = devices
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.handlers = []
        self.observers = []
        for device in devices:
            input_file_path = os.path.join(self.tmp_dir, f"input_{device}.{self.struct_format}")
            output_file_path = os.path.join(self.tmp_dir, f"output_{device}.pt")
            event_handler = SingleGPUInferenceHandler(
                input_file_path = input_file_path,
                output_file_path = output_file_path,
                model = self.model,
                device = f"cuda:{device}",
                batch_size = self.batch_size,
                num_workers = self.num_workers,
            )
            self.handlers.append(event_handler)
            observer = Observer()
            self.observers.append(observer)
            observer.schedule(event_handler, self.tmp_dir, recursive = False)
            observer.start()
    
    @override
    def run_inference(
        self,
        atoms_list: list[Atoms],
    ) -> list[dict[str, torch.Tensor]]:
        num_devices = len(self.devices)
        sizes = [len(atoms) for atoms in atoms_list]
        sorted_indices = np.argsort(sizes).tolist()
        distributed_data = [[] for _ in range(num_devices)]
        indices_map = [[] for _ in range(num_devices)]
        for i in range(len(atoms_list)):
            distributed_data[i % num_devices].append(atoms_list[sorted_indices[i]])
            indices_map[i % num_devices].append(sorted_indices[i])
            
        for i in range(num_devices):
            input_file_path = os.path.join(self.tmp_dir, f"input_{self.devices[i]}.{self.struct_format}")
            if self.struct_format == "npz":
                write_to_npz(distributed_data[i], input_file_path)
            else:
                write(input_file_path, distributed_data[i])
        
        ## wait for the inference to finish
        predictions = [None] * len(atoms_list)
        all_finished = False
        while not all_finished:
            all_finished = True
            for i in range(num_devices):
                output_file_path = os.path.join(self.tmp_dir, f"output_{self.devices[i]}.pt")
                if not os.path.exists(output_file_path):
                    all_finished = False
                    time.sleep(0.1)
                    break
        for i in range(num_devices):
            output_file_path = os.path.join(self.tmp_dir, f"output_{self.devices[i]}.pt")
            pred = torch.load(output_file_path)
            for j in range(len(pred)):
                predictions[indices_map[i][j]] = pred[j]
        os.system(f"rm -rf {self.tmp_dir}/*")
        predictions = cast(list[dict[str, torch.Tensor]], predictions)
        return predictions
    
    @override
    def exit(self):
        for observer in self.observers:
            observer.stop()
            observer.join()
        os.system(f"rm -rf {self.tmp_dir}/*")
        
class ParallizedInferenceDDP(ParallizedInferenceBase):
    def __init__(
        self,
        *,
        ckpt_path: str,
        properties: list[str],
        devices: list[int],
        batch_size: int,
        num_workers: int,
        using_partition: bool,
        inference_mode: bool = False,
        tmp_dir: str | None = None,
        input_filename: str = "input.npz",
        output_filename: str = "output.pt",
    ):
        super().__init__()
        if tmp_dir is not None:
            os.makedirs(tmp_dir, exist_ok=True)
            os.system(f"rm -rf {tmp_dir}/*")
        self.tmp_dir = tempfile.mkdtemp(dir=tmp_dir)
        os.system(f"rm -rf {self.tmp_dir}/*")
        # print(f"Temporary directory: {self.tmp_dir}")
        self.input_filename = input_filename
        self.output_filename = output_filename
        
        cuda_env = os.environ.copy()
        cuda_env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
        scripts_path = os.path.dirname(os.path.abspath(__file__))
        scripts_path = os.path.join(scripts_path, "multi_gpu_ddp.py")
        self.cmd = [
            "torchrun",
            "--nproc_per_node", str(len(devices)),
            "--nnodes", "1",
            scripts_path,
            "--ckpt_path", ckpt_path,
            "--workspace", self.tmp_dir,
            "--input_filename", self.input_filename,
            "--output_filename", self.output_filename,
            "--properties", ",".join(properties),
            "--devices", ",".join(map(str, devices)),
            "--batch_size", str(batch_size),
        ]
        if num_workers is not None:
            self.cmd.append("--num_workers")
            self.cmd.append(str(num_workers))
        if using_partition: self.cmd.append("--using_partition")
        if inference_mode:  self.cmd.append("--inference_mode")
        self.proc = subprocess.Popen(self.cmd, env=cuda_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    @override
    def run_inference(self, atoms_list: list[Atoms]) -> list[dict[str, torch.Tensor]]:
        try:
            # clean everything
            os.system(f"rm -rf {self.tmp_dir}/*")
                
            # write input and send new input signal
            inp_path = os.path.join(self.tmp_dir, self.input_filename)
            write_to_npz(atoms_list, inp_path)
            job_new_input_signal(self.tmp_dir)

            # wait for the inference to finish or fail
            finish = os.path.join(self.tmp_dir, JOB_FINISH_SIGNAL)
            fail   = os.path.join(self.tmp_dir, JOB_FAILURE_SIGNAL)
            t0 = time.time()
            while True:
                ## check if the process is finished or failed
                if os.path.exists(fail):
                    msg = open(fail).read()
                    raise RuntimeError(f"Remote inference failed: {msg}")
                if os.path.exists(finish):
                    break
                
                if time.time() - t0 > 600:
                    raise Warning(f"ParallizedInferenceDDP takes more than 10 minutes to finish, this normally indicates a deadlock.")
                
                time.sleep(0.02)

            # read output
            out_path = os.path.join(self.tmp_dir, self.output_filename)
            results = torch.load(out_path)

            # clean all signals and input/output files
            for fn in (finish, fail, inp_path, out_path):
                try: os.remove(fn)
                except: pass

            return results
        except Exception as e:
            self.exit()
            raise e

    @override
    def exit(self):
        exit_signal(self.tmp_dir)
        self.proc.wait(timeout=10)
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        