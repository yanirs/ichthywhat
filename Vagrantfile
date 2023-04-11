# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/jammy64"

  config.vm.provider "virtualbox" do |vb|
    vb.memory = "4096"
  end

  # FastAPI default port
  config.vm.network "forwarded_port", guest: 8000, host: 9300, host_ip: "127.0.0.1"
  # Streamlit default port
  config.vm.network "forwarded_port", guest: 8501, host: 9301, host_ip: "127.0.0.1"

  # Image parent path (see README.md; disabled by default)
  config.vm.synced_folder "/path/to/img", "/ichthywhat-pics/img", type: "rsync", disabled: true

  config.vm.provision "shell", name: "apt dependencies", inline: <<-SHELL
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
      python3-pip \
      zlib1g-dev
  SHELL

  config.vm.provision "shell", name: "poetry env", privileged: false, inline: <<-SHELL
    set -e
    cd /vagrant
    pip install poetry==1.4.1
    poetry install
    echo "Running the CLI to verify everything works"
    poetry run ichthywhat --help
  SHELL

  config.vm.provision "shell", name: "run servers", privileged: false, run: "always", inline: <<-SHELL
    echo "Running the FastAPI & Streamlit apps in screens"
    echo "FastAPI is on http://localhost:9300/ and Streamlit is on http://localhost:9301/"
    echo "In 'vagrant ssh', attach with 'screen -r api' or 'screen -r streamlit'"
    cd /vagrant
    screen -dmS api bash -c "poetry run uvicorn --reload --host 0.0.0.0 ichthywhat.api:api"
    screen -dmS streamlit bash -c "poetry run streamlit run ichthywhat/app.py"
  SHELL
end
