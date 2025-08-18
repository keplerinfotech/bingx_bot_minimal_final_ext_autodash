import os
import shutil

# List of files to delete
files_to_delete = [
    "README copy.md",
    "README copy 2.md",
    "multi_venue copy.py",
    "run_bingx_bot copy.command",
    "scripts/convert_recorder_multi copy.py",
    "scripts/dashboard copy.py",
    "scripts/run_final_replay copy.py",
    "scripts/run_final_replay copy 2.py",
]

# List of folders to delete
folders_to_delete = [
    ".dupe_backups",
    ".merge_conflicts",
]


def delete_files(files):
    for file in files:
        if os.path.isfile(file):
            os.remove(file)
            print(f"Deleted file: {file}")


def delete_folders(folders):
    for folder in folders:
        if os.path.isdir(folder):
            shutil.rmtree(folder)
            print(f"Deleted folder: {folder}")


if __name__ == "__main__":
    delete_files(files_to_delete)
    delete_folders(folders_to_delete)
    print("Cleanup complete.")
