import sys


if __name__ == "__main__":
    path = '../handout/tangshi.txt'
    path = sys.argv[1]

    print('Preparing csv data...')

    poetry = []
    with open(path, 'r') as f:
        poem = ''
        for line in f:
            if len(line) <= 1:
                poetry.append(poem)
                poem = ''
            else:
                poem += line.strip('\n')

    length = []
    for poem in poetry:
        length.append(len(poem))
    print('Max length of poem:', max(length))
    print('Num of poems:', len(poetry))

    with open('tangshi.csv', 'w', encoding='utf-8') as f:
        f.write('正文'+'\n')
        for poem in poetry:
            f.write(poem + '\n')
        f.writelines(poetry)
