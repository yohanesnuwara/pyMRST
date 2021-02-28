def setup():
  import subprocess

  def install(name):
    subprocess.call(['apt', 'install', name])
  
  def clone(url):
    subprocess.call(['git', 'clone', url])
  
  # Install octave and cloning repositories
  install('octave')
  clone('https://bitbucket.org/mrst/mrst-core.git')
  clone('https://bitbucket.org/mrst/mrst-autodiff.git')
  clone('https://github.com/yohanesnuwara/reservoir_datasets')
  clone('https://github.com/yohanesnuwara/pyMRST')
  
