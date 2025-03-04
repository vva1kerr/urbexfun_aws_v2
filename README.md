# SETUP COMMANDS

* `sudo apt update`
* `sudo apt-get update`
* `sudo apt upgrade -y`
* `sudo apt install git curl unzip tar make sudo vim wget -y`
* `git clone "Your-repository"`
* `sudo apt install python3 python3-pip`
* `sudo apt install python-is-python3`
* `sudo apt install python3.12-venv`
* `sudo apt install -y libgl1 mesa-utils xvfb`
* `sudo apt-get install -y gdal-bin libgdal-dev`
* `sudo apt-get install gmt gmt-dcw gmt-gshhg`
* `cd "Your-repository"`
* `apt install python<version>-venv` # if on Debian/Ubuntu
* `python -m venv venv`
* `source venv/bin/activate`
* `pip3 install -r requirements.txt`
* `python3 -m streamlit run <app>.py`
* `sudo <path>/venv/bin/python -m streamlit run Home.py --server.port 80`
* `nohup python3 -m streamlit run <app>.py --server.port 80`
### SETUP HTTP ACCESS
#### Install nginx
*  `sudo apt-get install nginx -y`
#### Create nginx
Clear the current configuration of nginx
* `sudo sh -c 'echo -n > /etc/nginx/sites-available/default'`
Open the nginx config file
* `sudo vi /etc/nginx/sites-available/default`
Then paste the following
```c
map $http_upgrade $connection_upgrade {
    default upgrade;
    '' close;
}

upstream backend {
    server 127.0.0.1:8501;
    keepalive 64;
}

server {
    listen 80;
    server_name _;
    location / {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```
Restart your nginx server
* `sudo service nginx restart`

# …or create a new repository on the command line

* echo "# blankslate" >> README.md
* `git init`
* `git add README.md`
* `git commit -m "first commit"`
* `git branch -M main`
* `git remote add origin git@github.com:vva1kerr/blankslate.git`
* `git push -u origin main`

## …or push an existing repository from the command line

* `git remote add origin git@github.com:vva1kerr/blankslate.git`
* `git branch -M main`
* `git push -u origin main`

# CONNECT TO EC2 INSTANCE (in WSL)

* `cd ~/.ssh`
* `ssh -i "aws_urbexfun.pem" <user>@ec2-<ip>.<region>.compute.amazonaws.com`
