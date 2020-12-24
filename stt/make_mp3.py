import subprocess
import sys
from datetime import datetime


def link_to_mp3(address):
    filename = datetime.now().strftime('%Y%m%d%H%M%S%f')
    # filename = 'audio'
    command = f'youtube-dl --extract-audio --audio-format mp3 --o result/{filename}.mp3 ' + address
    print(command)
    subprocess.run(command, shell=True)
    return f'result/{filename}.mp3'


def file_to_mp3(title):
    file_title = title.split('.')
    file_title = file_title[:-1]
    file_title = '.'.join(file_title)

    command = 'ffmpeg -i "' + title + '" -ar 16000 -ac 1 -f mp3 "' + file_title + '.mp3"'
    subprocess.run(command, shell=True)
    return file_title + '.mp3'


def make_mp3(method: str, link: str):
    if method == 'link':
        return link_to_mp3(link)
    elif method == 'file':
        return file_to_mp3(link)


if __name__ == "__main__":
    make_mp3(sys.argv[1], sys.argv[2])
