#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import yaml

def run_command(command, shell=False):
    """Run a shell command and print its output."""
    try:
        result = subprocess.run(command, shell=shell, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(command) if not shell else command}")
        print(e.stderr)
        sys.exit(e.returncode)

def export_env(env_name, yaml_filename, requirements_filename):
    """
    Export the Conda environment to a YAML file and pip packages to a requirements.txt file.
    """
    print(f"Exporting conda environment '{env_name}' to '{yaml_filename}'...")
    # Export the conda environment to YAML.
    run_command(["conda", "env", "export", "--name", env_name, "--file", yaml_filename])
    print("Conda environment exported successfully!")

    print(f"\nExporting pip packages in environment '{env_name}' to '{requirements_filename}'...")
    # Use conda run to export pip packages
    command = f"conda run -n {env_name} pip freeze > {requirements_filename}"
    run_command(command, shell=True)
    print("Pip requirements exported successfully!")

def import_env_advanced(yaml_filename, requirements_filename):
    """
    Create a conda environment using settings from the YAML file.
    The script extracts the environment name, Python version, and conda packages from the YAML file,
    creates the environment, installs the Conda packages, and then installs pip packages from the requirements file.
    """
    if not os.path.exists(yaml_filename):
        print(f"YAML file '{yaml_filename}' not found.")
        sys.exit(1)
    
    # Load YAML file
    with open(yaml_filename, 'r') as f:
        try:
            data = yaml.safe_load(f)
        except Exception as e:
            print(f"Failed to parse YAML file: {e}")
            sys.exit(1)
    
    env_name = data.get("name")
    if not env_name:
        print("Environment name not found in YAML file.")
        sys.exit(1)
    
    dependencies = data.get("dependencies", [])
    python_version = None
    conda_packages = []
    for dep in dependencies:
        if isinstance(dep, str):
            if dep.startswith("python="):
                # Expect a format like python=3.8
                python_version = dep.split("=")[1]
            else:
                conda_packages.append(dep)
        elif isinstance(dep, dict):
            # This typically contains pip packages; we'll handle pip separately.
            continue

    if not python_version:
        print("Python version not specified in YAML file; please ensure it includes an entry like 'python=3.x'.")
        sys.exit(1)
    
    print(f"Creating conda environment '{env_name}' with Python {python_version}...")
    run_command(["conda", "create", "--name", env_name, f"python={python_version}", "-y"])
    
    if conda_packages:
        print(f"Installing conda packages: {', '.join(conda_packages)} ...")
        run_command(["conda", "install", "--name", env_name] + conda_packages + ["-y"])
    else:
        print("No additional conda packages found in the YAML file.")

    # Install pip packages from the separate requirements file, if it exists.
    if os.path.exists(requirements_filename):
        print(f"Installing pip packages from '{requirements_filename}'...")
        run_command(f"conda run -n {env_name} pip install -r {requirements_filename}", shell=True)
    else:
        print(f"No requirements file '{requirements_filename}' found. Skipping pip installs.")
    
    print("Conda environment imported and set up successfully!")

def main():
    parser = argparse.ArgumentParser(description="Automate export/import of Conda environments and pip requirements")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command help")

    # Export subcommand.
    export_parser = subparsers.add_parser("export", help="Export the current conda environment")
    export_parser.add_argument("--env", type=str, required=True, help="Name of the conda environment to export")
    export_parser.add_argument("--yaml", type=str, default="environment.yml", help="Output YAML filename (default: environment.yml)")
    export_parser.add_argument("--req", type=str, default="requirements.txt", help="Output requirements filename (default: requirements.txt)")

    # Import subcommand.
    import_parser = subparsers.add_parser("import", help="Import a conda environment from a YAML file and install pip requirements")
    import_parser.add_argument("--yaml", type=str, default="environment.yml", help="YAML file to import (default: environment.yml)")
    import_parser.add_argument("--req", type=str, default="requirements.txt", help="Requirements file for pip packages (default: requirements.txt)")

    args = parser.parse_args()

    if args.command == "export":
        export_env(args.env, args.yaml, args.req)
    elif args.command == "import":
        import_env_advanced(args.yaml, args.req)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
