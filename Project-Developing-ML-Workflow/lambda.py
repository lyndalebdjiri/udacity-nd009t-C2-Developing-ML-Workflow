import json
import boto3
import base64
import sagemaker
from sagemaker.serializers import IdentitySerializer


def lambda_handler_serialize(event, context):
    s3 = boto3.client('s3')
    key = event['s3_key']
    bucket = event['s3_bucket']
    s3.download_file(bucket, key, '/tmp/image.png')
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }


def lambda_handler_classify(event, context):
    ENDPOINT = "your-endpoint-name"
    image = base64.b64decode(event['image_data'])
    predictor = sagemaker.predictor.Predictor(endpoint_name=ENDPOINT)
    predictor.serializer = IdentitySerializer("image/png")
    inferences = predictor.predict(image)
    event["inferences"] = inferences.decode('utf-8')
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }


def lambda_handler_filter(event, context):
    THRESHOLD = .93
    inferences = json.loads(event['inferences'])
    meets_threshold = any(i > THRESHOLD for i in inferences)
    if meets_threshold:
        pass
    else:
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
