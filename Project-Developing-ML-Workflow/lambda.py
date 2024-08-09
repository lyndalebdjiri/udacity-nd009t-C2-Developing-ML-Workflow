import json
import boto3
import base64
import sagemaker
from sagemaker.serializers import IdentitySerializer


s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""

    # Get the s3 address from the Step Function event input
    key = event['s3_key']
    bucket = event['s3_bucket']

    # Download the data from s3 to /tmp/image.png
    s3.download_file(bucket, key, '/tmp/image.png')

    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    return {
        "statusCode": 200,
        "image_data": image_data.decode('utf-8'),
        "s3_bucket": bucket,
        "s3_key": key,
        "inferences": []
    }




# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2024-08-09-03-50-35-740"

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event['image_data'])

    # Instantiate a Predictor
    predictor = sagemaker.predictor.Predictor(endpoint_name=ENDPOINT)

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")

    # Make a prediction:
    inferences = predictor.predict(image)

    # Converting the inferences to a list of floats
    inferences = json.loads(inferences.decode('utf-8'))

    # Converting the inferences list to a JSON string
    inferences_str = json.dumps(inferences)

    # We return the data back to the Step Function 
    return {
        "statusCode": 200,
        "image_data": event['image_data'],
        "s3_bucket": event['s3_bucket'],
        "s3_key": event['s3_key'],
        "inferences": inferences_str
    }



THRESHOLD = .93

def lambda_handler(event, context):

    # Grab the inferences from the event
    inferences = json.loads(event['inferences'])

    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = any(i > THRESHOLD for i in inferences)

    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
