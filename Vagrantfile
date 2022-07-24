# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/focal64"

  config.vm.provider "virtualbox" do |vb|
    vb.memory = "2048"
  end

  # Streamlit default port
  config.vm.network "forwarded_port", guest: 8501, host: 8501

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
      zlib1g-dev
  SHELL

  config.vm.provision "shell", name: "pyenv", privileged: false, inline: <<-SHELL
    curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/49fba599e872bc761858ea6f700271fb6dcb5a97/bin/pyenv-installer | bash
    echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.profile
    echo 'eval "$(pyenv init -)"' >> ~/.profile
  SHELL

  config.vm.provision "shell", name: "python & poetry env", privileged: false, inline: <<-SHELL
    cd /vagrant
    pyenv install
    pip install poetry==1.1.14
    poetry install
    echo "Running the CLI to verify everything works"
    poetry run ichthywhat --help
  SHELL

  config.vm.provision "shell", name: "run dev server", privileged: false, run: "always", inline: <<-SHELL
    echo "Running the Streamlit app in a screen. Open http://localhost:8501/ to view it."
    echo "In 'vagrant ssh', run 'screen -r' to attach to the screen and view the app's log."
    screen -dm bash -c "cd /vagrant && poetry run streamlit run ichthywhat/app.py"
  SHELL
end
