# Steps for the deployment to AWS for instance segmentation

## Step1: Train the YOLO model
Train the model for instance segmentation, after training we will be getting best.pt model. Export the "best.pt" file for inferencing in flask app.

## Step2: Build  a Flask API
## Step3: Deploy on AWS EC2
1. Launch an EC2 instance: 
- Choosing an appropriate instance(e.g. t3.medium for CPU or g4dn.xlarge for GPU).
- Configuring storage and security groups (open port for SSh and HTTP)
- SSH into the instance

        ssh -i "[/path/to/key.pem]" ubuntu@<your-ec2-ip>

2. Installing required packages:
- update the system:

        sudo apt update && sudo apt upgrade -y

- install dependencies:

        sudo apt update && sudo apt upgrade -y

3. Transfer files to EC2:
-  Use scp to transfer the flask app to the server

        scp -i path/to/key.pem -r your-app-folder ubuntu@<your-ec2-ip>:/home/ubuntu/app

4. Setup flask app on EC2:
- Navigate to the ap folder:

        scp -i path/to/key.pem -r your-app-folder ubuntu@<your-ec2-ip>:/home/ubuntu/app

- Create and activate a virtual environment:

        python3 -m venv venv
        source venv/bin/activate


- install dependencies:

        pip insall -r requirements.txt

5. Run Flask app with Gunicorn:

- Install Gunicorn

        pip install gunicorn

- Run the app: 

        gunicorn --bind 0.0.0.0:5000 app:app

## Step4: Configure Nginx



1. Edit Nginx Configuration:

        sudo apt update
        sudo apt install nginx

        sudo nano /etc/nginx/sites-available/flask-app

- Update /etc/nginx/sites-available/default:

        server {
        listen 80;

        location / {
            proxy_pass http://127.0.0.1:5000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }

        location /static/ {
            root /home/ubuntu/app;
        }

        location /uploads/ {
            root /home/ubuntu/app;
        }
    }

2. Restart Nginx:

        sudo systemctl restart nginx

Note: Make sure all the files and directories in the app directory have permission granted.

## Step 5: Automate App with Systemd

1. Create a Systemd Service:
- Create /etc/systemd/system/flask_app.service

        [Unit]
        Description=Gunicorn instance to serve Flask App
        After=network.target

        [Service]
        User=ubuntu
        Group=www-data
        WorkingDirectory=/home/ubuntu/app
        Environment="PATH=/home/ubuntu/app/venv/bin"
        ExecStart=/home/ubuntu/app/venv/bin/gunicorn --workers 3 --bind 127.0.0.1:5000 app:app

        [Install]
        WantedBy=multi-user.target

2. Start and Enable the service:

        sudo systemctl start flask_app
        sudo systemctl enable flask_app





