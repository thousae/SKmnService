import subprocess
import sys
from datetime import datetime


def link_to_wav(address):
    filename = datetime.now().strftime('%Y%m%d%H%M%S%f')
    filename_temp = filename + '_temp'
    command = f"youtube-dl --extract-audio --audio-format wav --o result/{filename_temp}.wav " + address
    subprocess.run(command, shell=True)
    command = f'ffmpeg -i "result/{filename_temp}.wav" -ar 16000 -ac 1 -f wav "result/{filename}.wav"'
    subprocess.run(command, shell=True)
    return "result/" + filename + ".wav"


def file_to_wav(title):
    file_title = title.split('.')
    file_title = file_title[:-1]
    file_title = '.'.join(file_title)

    command = f'ffmpeg -i "{title}" -ar 16000 -ac 1 -f wav "{file_title}.wav"'
    subprocess.run(command, shell=True)
    return file_title + '.wav'


def make_wav(method: str, link: str):
    if method == 'link':
        return link_to_wav(link)
    elif method == 'file':
        return file_to_wav(link)

if __name__ == "__main__":
    make_wav(sys.argv[1], sys.argv[2])
