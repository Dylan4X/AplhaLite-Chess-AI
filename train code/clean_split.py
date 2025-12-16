import chess.pgn
import os
import time
import multiprocessing
from multiprocessing import Process, cpu_count

# ================= CONFIGURATION =================

# Path to the directory containing source .pgn files
# Structure:
#   INPUT_ROOT/train/*.pgn
#   INPUT_ROOT/test/*.pgn
INPUT_ROOT = 'path/to/raw_data'

# Path where split files will be saved
OUTPUT_ROOT = 'path/to/dataset'

# Number of files per subdirectory to avoid filesystem inode limits
# (Recommended: 10000)
FILES_PER_SUBDIR = 10000


# ===============================================

def worker_process(file_list, output_base_dir, process_id):
    """
    Worker process to read PGN files and save each game as a separate file.
    """
    total_processed = 0
    start_time = time.time()

    for file_path in file_list:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as pgn_fh:
                while True:
                    # Read game from PGN
                    try:
                        game = chess.pgn.read_game(pgn_fh)
                    except Exception:
                        continue

                    if game is None:
                        break

                        # Determine subdirectory index to organize files
                    group_idx = total_processed // FILES_PER_SUBDIR
                    subdir_name = f"group_{process_id}_{group_idx}"
                    full_subdir_path = os.path.join(output_base_dir, subdir_name)

                    # Create subdirectory if it does not exist
                    if total_processed % FILES_PER_SUBDIR == 0:
                        os.makedirs(full_subdir_path, exist_ok=True)

                    # Generate output filename: game_{core_id}_{index}.pgn
                    filename = f"game_{process_id}_{total_processed}.pgn"
                    save_path = os.path.join(full_subdir_path, filename)

                    # Write single game to file
                    with open(save_path, 'w', encoding='utf-8') as f_out:
                        exporter = chess.pgn.FileExporter(f_out)
                        game.accept(exporter)

                    total_processed += 1

                    # Log progress every 5000 games
                    if total_processed % 5000 == 0:
                        elapsed = time.time() - start_time
                        speed = total_processed / elapsed if elapsed > 0 else 0
                        print(f"[Core {process_id}] Processed: {total_processed} | Speed: {speed:.0f} games/s")

        except Exception as e:
            print(f"[Error] Core {process_id} failed on {file_path}: {e}")

    print(f"[Core {process_id}] Finished. Total games: {total_processed}")


def process_dataset(dataset_type):
    """
    Orchestrate the splitting process for a specific dataset type (train/test).
    """
    input_dir = os.path.join(INPUT_ROOT, dataset_type)
    output_dir = os.path.join(OUTPUT_ROOT, dataset_type)

    print(f"--- Processing Dataset: {dataset_type} ---")

    if not os.path.exists(input_dir):
        print(f"[Warning] Input directory not found: {input_dir}")
        return

    # Gather all .pgn files
    all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".pgn")]
    if not all_files:
        print(f"[Warning] No .pgn files found in {input_dir}")
        return

    # Determine optimal number of processes
    total_cores = cpu_count()
    # Reserve 2 cores for system stability
    num_processes = max(1, total_cores - 2)
    num_processes = min(num_processes, len(all_files))

    # Distribute files among processes
    chunk_size = len(all_files) // num_processes + 1
    processes = []

    for i in range(num_processes):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(all_files))
        process_files = all_files[start:end]

        if process_files:
            p = Process(target=worker_process, args=(process_files, output_dir, i))
            processes.append(p)
            p.start()

    # Wait for completion
    for p in processes:
        p.join()


if __name__ == '__main__':
    # Process Training Data
    process_dataset('train')

    # Process Test Data
    process_dataset('test')

    print("--- All Tasks Completed ---")