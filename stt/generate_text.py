from google.cloud import storage
from google.cloud import speech
import subprocess
import os


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    # upload file to storage
    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )


def long_running_recognize(storage_uri):
    """
    Transcribe long audio file from Cloud Storage using asynchronous speech
    recognition

    Args:
      storage_uri URI for audio file in Cloud Storage, e.g. gs://[BUCKET]/[FILE]
    """

    print('stt start')

    client = speech.SpeechClient()

    # storage_uri = 'gs://cloud-samples-data/speech/brooklyn_bridge.raw'
    
    # encoding: audio file format
    # sample_rate_hertz: audio file hertz
    # language_code: language of audio. English - en_US, Korean - 'ko_KR'
    # enable_automatic_punctuation: automatic punctuation
    
    config = speech.RecognitionConfig(
        encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz = 16000,
        language_code = 'en_US',
        enable_automatic_punctuation = True
    )

    # audio file in storage
    audio = speech.RecognitionAudio(uri = storage_uri)

    # running
    operation = client.long_running_recognize(config = config, audio = audio)
    
    # take result
    response = operation.result()
    print('stt over')

    return response


def generate_text(filepath):
    
    # json_file_path = your json key file path
    # bucket_name = name of bucket for uploading file
    # storage_uri = storage uri for uploading file and running STT 
    
    json_file_path = os.getcwd() + '/stt/[json-file-name]'
    bucket_name = 'speechtotext_bin_bucket'
    storage_uri = 'gs://' + bucket_name + '/' + filepath.split("/")[-1]
    
    # setting
    # out_file = result file name
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=json_file_path
    out_file = filepath.split(".")[0] + '.txt'
    
    # upload file to storage and take result
    
    upload_blob(bucket_name, filepath, filepath.split("/")[-1])
    response = long_running_recognize(storage_uri)
    
    # make txt file
    
    f = open(filepath.split(".")[0] + '.txt', 'w')
    for result in response.results:
        text = result.alternatives[0].transcript
        f.write(text)
    f.close()

    return out_file


