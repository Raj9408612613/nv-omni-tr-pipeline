monito usage : 
# Install required packages
pip3 install psutil

# Make scripts executable
chmod +x gpu_monitor.py cpu_monitor.py combined_monitor.py

# Run individual monitors
python3 gpu_monitor.py
python3 cpu_monitor.py

# Or run combined monitor
python3 combined_monitor.py


# Running isaac lab on a EC2 instance
git clone

run setup_ec2_isaac.sh

run verify_setup.sh

run smmoke_tess.sh --full 

run isaaclab_run.sh
