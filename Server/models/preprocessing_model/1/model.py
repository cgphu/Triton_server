import numpy as np
import json
import cv2
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args['model_config'])
        

    def execute(self, requests):
        """
        This function is called when an inference request is made.
        It preprocesses the input image to convert it to [1, 3, 320, 320] dimensions.
        """
        responses = []

        for request in requests:
            # Extract input tensor from the request
          preprocessed_data=[]
          input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
          input_data = input_tensor.as_numpy()
          for origin_image in input_data:
            origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
            [height, width, _] = origin_image.shape
            length = max((height, width))
            image = np.zeros((length, length, 3), np.uint8)
            image[0:height, 0:width] = origin_image

            #Resize to the expected input size
            image = cv2.resize(image, (320,320 ))
            image = image.astype(np.float32)/ 255.0
            # print(image)
            # Convert format from NHWC to NCHW
            image = np.transpose(image, (2, 0, 1))

            # Add a new axis to represent the batch size (1)
            #image = np.expand_dims(image, axis=0)  # Shape becomes [1, 3, 320, 320]
            preprocessed_data.append(image)
          # Create output tensor with the preprocessed data
          preprocessed_data=np.array(preprocessed_data)
          output_tensor = pb_utils.Tensor("PREPROCESSED_OUTPUT", preprocessed_data)

          # Create an InferenceResponse containing the output tensor
          response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
          responses.append(response)

        return responses

    def finalize(self):
        """
        Cleanup function, called when the model is unloaded.
        """
        print("Image preprocessing model finalized.")