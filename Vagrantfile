# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/jammy64"

  config.vm.provider "virtualbox" do |vb|
    vb.memory = 8192
    vb.cpus = 4
  end

  # FastAPI default port
  config.vm.network "forwarded_port", guest: 8000, host: 9300, host_ip: "127.0.0.1"
  # Streamlit default port
  config.vm.network "forwarded_port", guest: 8501, host: 9301, host_ip: "127.0.0.1"
  # MLflow default port
  config.vm.network "forwarded_port", guest: 5000, host: 9302, host_ip: "127.0.0.1"
  # Jupyter notebook default port
  config.vm.network "forwarded_port", guest: 8888, host: 9303, host_ip: "127.0.0.1"

  # Image parent path (see README.md; disabled by default)
  config.vm.synced_folder "/path/to/img", "/ichthywhat-pics/img", type: "rsync", disabled: true

  config.vm.provision "shell", name: "apt-dependencies", inline: <<-SHELL
    apt-get update
    apt-get install -y \
      build-essential \
      libbz2-dev \
      libffi-dev \
      liblzma-dev \
      libncursesw5-dev \
      libreadline-dev \
      libsqlite3-dev \
      libssl-dev \
      pipx \
      podman \
      zlib1g-dev
  SHELL

  config.vm.provision "shell", name: "poetry", privileged: false, inline: <<-SHELL
    pipx install poetry==1.4.2
  SHELL

  config.vm.provision "shell", name: "python-dependencies", privileged: false, inline: <<-SHELL
    set -e
    cd /vagrant
    poetry install
    poetry run pre-commit install
    echo "Running the CLI to verify everything works"
    poetry run ichthywhat --help
  SHELL

  config.vm.provision "shell", name: "apt-upgrade", run: "always", inline: <<-SHELL
    apt-get update && apt-get upgrade -y
  SHELL

  config.vm.provision "shell", name: "run-servers", privileged: false, run: "never", inline: <<-SHELL
    echo "Running servers in screens:"
    echo " - FastAPI is on http://localhost:9300/ ('screen -r api')"
    echo " - Streamlit is on http://localhost:9301/ ('screen -r streamlit')"
    echo " - MLflow is on http://localhost:9302/ ('screen -r mlflow')"
    echo " - Jupyter notebook is on http://localhost:9303/ ('screen -r notebook')"
    cd /vagrant
    screen -dmS api bash -c "poetry run uvicorn --reload --host 0.0.0.0 ichthywhat.api:api"
    screen -dmS streamlit bash -c "poetry run streamlit run ichthywhat/app.py"
    screen -dmS mlflow bash -c "poetry run mlflow ui --host 0.0.0.0 --backend-store-uri sqlite:///mlruns.db"
    screen -dmS notebook bash -c "poetry run jupyter notebook --ip 0.0.0.0"
  SHELL
end
