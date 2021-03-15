
git clone https://github.com/nkinnaird/MangroveClassification.git
#git clone git@github.com:nkinnaird/MangroveClassification.git
git config --global user.email "nickkinn@bu.edu"
git config --global user.name "Nicholas Kinnaird"

pip install pyrsgis
pip install matplotlib_scalebar
pip install jupyterlab

ssh-keygen -t ed25519 -C "nickkinn@bu.edu"
eval `ssh-agent -s`
ssh-add
jupyter lab --ip=0.0.0.0 --port=8989 --allow-root & ssh -o StrictHostKeyChecking=no -R 80:localhost:8989 ssh.localhost.run


