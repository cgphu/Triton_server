# SIMPLE TRITON SERVER WITH ENSEMBLE MODEL

## Directory Structure
Client/

  images/

  predict/
  
  client.py
  
  Dockerfile
  
  requirement.txt

Server/
  
  models/

      preprocessing_model/
          1/
              model.py
          config.pbtxt

      yolo_onnx/
          1/
              model.onnx
          config.pbtxt
          labels.txt

      ensemble_model/
          1/
              <Empty Directory>
          config.pbtxt

  Dockerfile

README.md

## Installation
Instructions on how to install and run the project locally.

```bash
# Clone the repository
git clone https://github.com/cgphu/Triton_server.git

# Change directory to the project folder
cd project-name

# Create an triton server image with the needed libraries
cd Server
docker build -t your-image-name . 
docker run -it --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v "path_to_where_the_prj_saved/Triton_server/Server/models:/models" your-image-name:latest tritonserver --model-repository=/models
# example I saved it in the D disk and my image name is tritonserver-with-opencv:latest : 
# docker run -it --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v "D:/Triton_server/Server/models:/models" tritonserver-with-opencv:latest tritonserver --model-repository=/models 

# Create a environment image to run client in another terminal
cd Client
docker build -t your-image-name .
# get into image bin/bash and mount the Client folder to the app folder in image
docker run -it --rm -v "path_to_where_the_prj_saved/Triton_server/Client:/app" custom /bin/bash
# run the client.py to run the inference
python3 client.py
