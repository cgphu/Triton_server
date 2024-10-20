import os
import numpy as np
import cv2
import tritonclient.http as httpclient

# Get list of file_paths from folder to predict
image_files=[]
for file_name in (os.listdir('./images')):
  full_path = os.path.join('./images',file_name)
  image_files.append(full_path)
  
# Read these images and add a extra dims to fit the input of model
preprocess_images=[]
for image_path in image_files:
    origin_image=cv2.imread(image_path)
    if origin_image is None:
      raise Exception("Error: Image not loaded properly")
    image=np.expand_dims(origin_image,axis=0)
    preprocess_images.append(image)
    
# Connect to Triton Inference Server
triton_client = httpclient.InferenceServerClient(url="192.168.73.228:8000")

# Define the model name
model_name = "ensemble_model"
class_labels = ['PMS', 'defib', 'syringepump', 'ultrasound', 'ventilator', 'xray']
colors = np.random.uniform(0, 255, size=(len(class_labels), 3))
boxes_info=[]
class_ids_info=[]
scores_info=[]
threshold=0.5
for i in range(0,len(preprocess_images),4):
    batch_images=preprocess_images[i:i+4]
    #Create batch by stacking images
    batch_images_np=np.vstack(batch_images)
    # inputs=httpclient.InferInput('images', batch_images_np.shape, "FP32")
    inputs=httpclient.InferInput('RAW_INPUT', batch_images_np.shape, "UINT8")
    inputs.set_data_from_numpy(batch_images_np)

    outputs=httpclient.InferRequestedOutput('output0')

    # Send the request to Triton Server
    print("Batch ",i//4+1)
    response = triton_client.infer(model_name, inputs=[inputs], outputs=[outputs])
    # Get the output data from the response
    output_data = response.as_numpy('output0')
    output_data=np.transpose(output_data,(0,2,1))
    #Check the output of each image in batch
    for idx, output in enumerate(output_data):
        rows=output.shape[0]
        boxes_of_1image=[]
        scores_of_1image=[]
        class_ids_of_1image=[]
        for i in range(rows):
            classes_scores = output[i][4:]  # Assuming class scores are here
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            
            if maxScore >= threshold: #Check if maxScore >= threshold
                box = [
                    output[i][0] - (0.5 * output[i][2]),
                    output[i][1] - (0.5 * output[i][3]),
                    output[i][2],
                    output[i][3],
                ]
                boxes_of_1image.append(box)
                scores_of_1image.append(maxScore)
                class_ids_of_1image.append(maxClassIndex)
        boxes_info.append(boxes_of_1image)
        class_ids_info.append(class_ids_of_1image)
        scores_info.append(scores_of_1image)
# function draw bounding boxes from the output info
def draw_boxes(image_files, boxes_info,class_ids_info,scores_info,predict_folder):
    print("drawing boxes ...")
    # Load the image
    for i in range(len(image_files)):
        image=cv2.imread(image_files[i])
        height, width, channels = image.shape
        length=max(height, width)
        x_scale=length/320
        y_scale=length/320

        # Apply Non-Maximum Suppression (NMS)
        result_boxes = cv2.dnn.NMSBoxes(boxes_info[i], scores_info[i], threshold, 0.4)  #0.5 is threshold
        detections=[]
        for idx in result_boxes:
            box = boxes_info[i][idx]
            left, top, width, height = box
            detection = {
                "class_id": class_ids_info[i][idx],
                "class_name": class_labels[class_ids_info[i][idx]],
                "confidence": scores_info[i][idx],
                "box": box,
            }
            detections.append(detection)
            label = f'ID: {class_ids_info[i][idx]}, {class_labels[class_ids_info[i][idx]]}, Conf: {scores_info[i][idx]:.2f}'
            color = colors[class_ids_info[i][idx]]
            cv2.rectangle(image, (int(left*x_scale), int(top*y_scale)), (int((left+width)*x_scale), int((top+height)*y_scale)), color, 2)
            cv2.putText(image, label, (int(left*x_scale-10),int(top*y_scale - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # Save the image
        os.makedirs('./'+predict_folder, exist_ok=True)
        file_path = image_files[i]
        file_name = os.path.basename(file_path)
        cv2.imwrite('./'+predict_folder+'/'+file_name, image)
# Call the function
draw_boxes(image_files,boxes_info,class_ids_info,scores_info,'predict')


