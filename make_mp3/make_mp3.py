import subprocess
import sys


def link_to_mp3(address):
    command = 'youtube-dl --extract-audio --audio-format mp3 --o ./result/%(title)s.%(ext)s ' + address
    command = command.split(' ')
    subprocess.run(command)


def file_to_mp3(title):
    file_title = title.split('.')
    file_title = file_title[:-1]
    file_title = '.'.join(file_title)

    command = 'ffmpeg -i "' + title + '" -ar 16000 -ac 1 -f mp3 "' + file_title + '.mp3"'
    subprocess.run(command)


def make_mp3(arg: str, link: str):
    if arg == 'link':
        link_to_mp3(link)
    elif arg == 'file':
        file_to_mp3(link)


if __name__ == "__main__":
    make_mp3(sys.argv[1], sys.argv[2])