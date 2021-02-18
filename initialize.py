"""
Initialize simulator
Install Octave and clone repos, takes ~3-5 mins.
"""
# Install Octave
!apt install -q octave

# Cloning repositories
!git clone https://bitbucket.org/mrst/mrst-core.git
!git clone https://yohanesnuwara@bitbucket.org/mrst/mrst-autodiff.git
!git clone https://github.com/yohanesnuwara/reservoir_datasets
!git clone https://github.com/yohanesnuwara/pyMRST
