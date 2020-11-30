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

    # Encoding of audio data sent. This sample sets this explicitly.
    # This field is optional for FLAC and WAV audio formats.
    
    #speech_context = speech.SpeechContext(phrases=[])

    config = speech.RecognitionConfig(
        encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz = 16000,
        language_code = 'en_US',
        enable_automatic_punctuation = True
        #,speech_contexts = [speech_context]
    )

    audio = speech.RecognitionAudio(uri = storage_uri)

    operation = client.long_running_recognize(config = config, audio = audio)
    
    response = operation.result()
    print('stt over')

    return response


def generate_text(filepath):
    json_file_path = os.getcwd() + '/speech-to-text-285502-9513cb3afbc8.json'
    bucket_name = 'speechtotext_bin_bucket'
    storage_uri = 'gs://' + bucket_name + '/' + filepath.split("/")[-1]
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=json_file_path
    out_file = filepath.split(".")[0] + '.txt'
    upload_blob(bucket_name, filepath, filepath.split("/")[-1])
    response = long_running_recognize(storage_uri)
    
    f = open(filepath.split(".")[0] + '.txt', 'w')
    for result in response.results:
        text = result.alternatives[0].transcript
        f.write(text)
    f.close()

    return out_file


