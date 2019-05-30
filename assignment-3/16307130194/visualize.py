from matplotlib import pyplot as plt


def extract(log_path, slice=1):
    loss = []
    with open(log_path, 'r') as f:
        for line in f:
            tmp = line.split('(')
            if tmp[0] == 'tensor':
                tmp = tmp[1].split(',')
                loss.append(float(tmp[0]))

    return loss[2:min(len(loss), 40000):slice]


def draw():
    adam = extract('adam_log00.txt')
    sgd = extract('sgd_log21.txt')
    sgdm = extract('sgdm_log11.txt')
    adagrad = extract('adagrad_log11.txt')
    adadelta = extract('adadelta_log21.txt')

    plt.subplot(121)
    plt.plot(adam, label='Adam')
    plt.plot(sgd, label='SGD')
    plt.plot(sgdm, label='SGD with momentum')
    plt.plot(adagrad, label='Adagrad')
    plt.plot(adadelta, label='Adadelta')

    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('optimizer comparison')
    plt.legend()

    slice = 200
    adam = extract('adam_log00.txt', slice=slice)
    sgd = extract('sgd_log21.txt', slice=slice)
    sgdm = extract('sgdm_log11.txt', slice=slice)
    adagrad = extract('adagrad_log11.txt', slice=slice)
    adadelta = extract('adadelta_log21.txt', slice=slice)

    plt.subplot(122)
    plt.plot(adam, label='Adam')
    plt.plot(sgd, label='SGD')
    plt.plot(sgdm, label='SGD with momentum')
    plt.plot(adagrad, label='Adagrad')
    plt.plot(adadelta, label='Adadelta')

    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('smoothed optimizer comparison')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    draw()
