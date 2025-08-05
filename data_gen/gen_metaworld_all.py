import os
import shutil

import subprocess

def load_task_data(base_path, device_ids, sample_pt_num=400, num_samples=1, data_buffer_path=None):

    model_params = {
        'grounded_sam': {},  # Add any specific parameters for GoundedSam2
        'tracker': {}        # Add any specific parameters for PointTracker
    }

    sequences = []
    task_names = []
    for task_name in sorted(os.listdir(base_path)):
        task_path = os.path.join(base_path, task_name)
        task_name = task_name.replace("-", " ")
        task_names.append(task_name)
        if os.path.isdir(task_path):
            camera_name = "corner"
            camera_path = os.path.join(task_path, camera_name)
            if os.path.isdir(camera_path):
                for sequence_name in sorted(os.listdir(camera_path)):
                    sequence_path = os.path.join(camera_path, sequence_name)
                    if os.path.isdir(sequence_path):
                        sequences.append((task_name, sequence_name, sequence_path))

    print(f"Total tasks: {len(set(os.listdir(base_path)))}")
    print(f"Total sequences: {len(sequences)}")

    
    processes = []
    task_idx = {task_name: 0 for task_name in task_names}
    for i, (task_name, sequence_name, sequence_path) in enumerate(sequences):
        env = os.environ.copy()
        device_id = device_ids[i % len(device_ids)]
        env["CUDA_VISIBLE_DEVICES"] = str(device_id)
        global_index_start = task_idx[task_name]
        sequence_info = (
            data_buffer_path,
            sequence_path, 
            sample_pt_num, 
            num_samples, 
            model_params, 
            global_index_start,
            task_name,
        )
        task_idx[task_name] += num_samples
        
        process = subprocess.Popen(
            [
                "python", 
                "-c",
                f"from data_gen.gen_metaworld import process_sequence; process_sequence({sequence_info})",
            ],
            env=env,
        )
        processes.append(process)
        
        if i % len(device_ids) == len(device_ids) - 1:
            for process in processes:
                process.wait()
            processes = []


if __name__ == "__main__":
    # Example configuration
    base_path = "data/metaworld_original"
    device_ids = [0,1,2,3,4,5,6,7]  # GPU IDs to use
    
    sample_pt_num = 4000 # Number of points to sample from each frame
    num_samples = 20  # Number of samples per sequence
    data_buffer_path = "data/metaworld"  # Path to Zarr data buffer
    
    clear_buffer = True
    
    if clear_buffer:
        if os.path.exists(data_buffer_path):
            shutil.rmtree(data_buffer_path) 
            print("Old data buffer removed.")
            
    # Run the data loading process
    load_task_data(
        base_path=base_path,
        device_ids=device_ids,
        sample_pt_num=sample_pt_num,
        num_samples=num_samples,
        data_buffer_path=data_buffer_path
    )

    print("Data loading completed and stored in Zarr data buffer.")