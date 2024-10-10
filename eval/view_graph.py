import subprocess

def run_script(script_path):
    subprocess.run(["python", script_path], check=True)

def main():
    scripts = [
        'scripts/extract_weights.py',
        'scripts/compute_distances.py',
        'scripts/define_morphisms.py',
        'visualization/visualize_weight_space.py'
    ]
    
    for script in scripts:
        print(f"Running {script}...")
        run_script(script)
        print(f"Completed {script}\n")

if __name__ == "__main__":
    main()