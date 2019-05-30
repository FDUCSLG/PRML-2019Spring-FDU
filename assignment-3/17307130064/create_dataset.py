import os
import json
import re
import git


def gitclone():
    repo = git.Repo.clone_from(url='https://github.com/jackeyGao/chinese-poetry.git', to_path='./dataset')

def handle_json(filename):
    file = open(filename, 'rb')
    poetry_list = file.read()
    poetry_list = json.loads(poetry_list)

    content_list = []
    for poetry in poetry_list:
        content = ''
        for sentence in poetry.get("paragraphs"):
            content += sentence
        content_list.append(content)
    return content_list

def process_content(content):
    result, _ = re.subn('（.*）', '', content)
    result, _ = re.subn('[.*]', '', result)
    result, _ = re.subn('{.*}', '', result)
    result, _ = re.subn('。。', '。', result)
    RESULT = ''
    for res in result:
        RESULT += res
    if len(RESULT) >= 12:
        return RESULT

def create_POETRY_LIST():
    POETRY_LIST = []
    folder_path = './dataset/json/'
    for filename in os.listdir(folder_path):
        if not filename.startswith('poet.tang'):
            continue
        content_list = handle_json(folder_path + filename)
        for content in content_list:
            content_done = process_content(content)
            if content_done is not None and len(content_done) <= 48:
                POETRY_LIST.append(content_done)

    return POETRY_LIST