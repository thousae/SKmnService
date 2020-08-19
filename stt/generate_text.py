import subprocess

apikey = 'VeZTw8-B4TfmOdLXH3H1517CkBjP9AG8dHeELrPS_2jV'
url = 'https://api.kr-seo.speech-to-text.watson.cloud.ibm.com/instances/28994288-68ad-443d-aa0c-1444836198ba'

def generate_text(filepath):
    out_file = filepath.split(".")[0] + '.txt'
    command = f"""curl -X POST -u "apikey:{apikey}" \\
--header "Content-Type: audio/mp3" \\
--data-binary @{filepath} > {out_file} \\
"{url}/v1/recognize" """
    print(command)
    subprocess.run(command, shell=True)
    return out_file
