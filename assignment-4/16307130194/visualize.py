from matplotlib import pyplot as plt


def extract(log_path):
    loss = []
    acc = []
    with open(log_path, 'r') as f:
        for line in f:
            loss_split = line.split('loss: ')
            if len(loss_split) >= 2:
                time_split = loss_split[1].split(' time')
                if len(time_split) >= 2:
                    loss.append(float(time_split[0]))
            acc_split = line.split('acc=')
            if len(acc_split) >= 2:
                acc.append(float(acc_split[1]))

    # return loss[2:min(len(loss), 40000):slice]
    return loss[::10], acc[::1]


def draw0():
    cnn_loss, cnn_acc = extract('cnn_log00.txt')
    lstm_loss, lstm_acc = extract('lstm_log00.txt')
    rcnn_loss, rcnn_acc = extract('rcnn_log00.txt')

    plt.subplot(121)
    plt.plot(cnn_loss, label='CNN')
    plt.plot(lstm_loss, label='LSTM')
    plt.plot(rcnn_loss, label='RCNN')
    plt.xlabel('100 steps')
    plt.ylabel('loss')
    plt.title('loss comparison')
    plt.legend()

    plt.subplot(122)
    plt.plot(cnn_acc[:-2], label='CNN, dev: %.2f, test: %.2f' % (cnn_acc[-2] * 100, cnn_acc[-1] * 100))
    plt.plot(lstm_acc[:-2], label='LSTM, dev: %.2f, test: %.2f' % (lstm_acc[-2] * 100, lstm_acc[-1] * 100))
    plt.plot(rcnn_acc[:-2], label='RCNN, dev: %.2f, test: %.2f' % (rcnn_acc[-2] * 100, rcnn_acc[-1] * 100))
    plt.xlabel('100 steps')
    plt.ylabel('accuracy')
    plt.title('accuracy comparison')
    plt.legend()

    plt.show()

def draw1():
    lstm_loss, lstm_acc = extract('lstm_log00.txt')
    lstm_mp_loss, lstm_mp_acc = extract('lstm_log01.txt')
    rcnn_loss, rcnn_acc = extract('rcnn_log00.txt')

    plt.subplot(121)
    plt.plot(lstm_loss, label='LSTM')
    plt.plot(lstm_mp_loss, label='LSTM_paxpool')
    plt.plot(rcnn_loss, label='RCNN')
    plt.xlabel('100 steps')
    plt.ylabel('loss')
    plt.title('loss comparison')
    plt.legend()

    plt.subplot(122)
    plt.plot(lstm_acc[:-2], label='LSTM, dev: %.2f, test: %.2f' % (lstm_acc[-2] * 100, lstm_acc[-1] * 100))
    plt.plot(lstm_mp_acc[:-2], label='LSTM_paxpool, dev: %.2f, test: %.2f' % (lstm_mp_acc[-2] * 100, lstm_mp_acc[-1] * 100))
    plt.plot(rcnn_acc[:-2], label='RCNN, dev: %.2f, test: %.2f' % (rcnn_acc[-2] * 100, rcnn_acc[-1] * 100))
    plt.xlabel('100 steps')
    plt.ylabel('accuracy')
    plt.title('accuracy comparison')
    plt.legend()

    plt.show()

def draw2():
    rnn_loss, rnn_acc = extract('rnn_log00.txt')
    lstm_loss, lstm_acc = extract('lstm_log00.txt')

    plt.subplot(121)
    plt.plot(rnn_loss, label='RNN')
    plt.plot(lstm_loss, label='LSTM')
    plt.xlabel('100 steps')
    plt.ylabel('loss')
    plt.title('loss comparison')
    plt.legend()

    plt.subplot(122)
    plt.plot(rnn_acc[:-2], label='RNN, dev: %.2f, test: %.2f' % (rnn_acc[-2] * 100, rnn_acc[-1] * 100))
    plt.plot(lstm_acc[:-2], label='LSTM, dev: %.2f, test: %.2f' % (lstm_acc[-2] * 100, lstm_acc[-1] * 100))
    plt.xlabel('100 steps')
    plt.ylabel('accuracy')
    plt.title('accuracy comparison')
    plt.legend()

    plt.show()

def draw3():
    cnn_loss, cnn_acc = extract('cnn_log00.txt')
    cnnw_loss, cnnw_acc = extract('cnnw2v_log00.txt')

    plt.subplot(121)
    plt.plot(cnn_loss, label='CNN')
    plt.plot(cnnw_loss, label='CNN_w2v')
    plt.xlabel('100 steps')
    plt.ylabel('loss')
    plt.title('loss comparison')
    plt.legend()

    plt.subplot(122)
    plt.plot(cnn_acc[:-2], label='CNN, dev: %.2f, test: %.2f' % (cnn_acc[-2] * 100, cnn_acc[-1] * 100))
    plt.plot(cnnw_acc[:-2], label='CNN_w2v, dev: %.2f, test: %.2f' % (cnnw_acc[-2] * 100, cnnw_acc[-1] * 100))
    plt.xlabel('100 steps')
    plt.ylabel('accuracy')
    plt.title('accuracy comparison')
    plt.legend()

    plt.show()


def draw4():
    cnn_loss, cnn_acc = extract('cnnw2v_log00.txt')
    cnnw_loss, cnnw_acc = extract('cnnw2v_log01.txt')

    plt.subplot(121)
    plt.plot(cnn_loss, label='window:1')
    plt.plot(cnnw_loss, label='window:64')
    plt.xlabel('100 steps')
    plt.ylabel('loss')
    plt.title('loss comparison')
    plt.legend()

    plt.subplot(122)
    plt.plot(cnn_acc[:-2], label='window:1, dev: %.2f, test: %.2f' % (cnn_acc[-2] * 100, cnn_acc[-1] * 100))
    plt.plot(cnnw_acc[:-2], label='window:64, dev: %.2f, test: %.2f' % (cnnw_acc[-2] * 100, cnnw_acc[-1] * 100))
    plt.xlabel('100 steps')
    plt.ylabel('accuracy')
    plt.title('accuracy comparison')
    plt.legend()

    plt.show()


def draw5():
    cnn_loss, cnn_acc = extract('rcnn_log00.txt')
    cnnw_loss, cnnw_acc = extract('rcnn_log01.txt')

    plt.subplot(121)
    plt.plot(cnn_loss, label='num_layer:1')
    plt.plot(cnnw_loss, label='num_layer:2')
    plt.xlabel('100 steps')
    plt.ylabel('loss')
    plt.title('loss comparison')
    plt.legend()

    plt.subplot(122)
    plt.plot(cnn_acc[:-2], label='num_layer:1, dev: %.2f, test: %.2f' % (cnn_acc[-2] * 100, cnn_acc[-1] * 100))
    plt.plot(cnnw_acc[:-2], label='num_layer:2, dev: %.2f, test: %.2f' % (cnnw_acc[-2] * 100, cnnw_acc[-1] * 100))
    plt.xlabel('100 steps')
    plt.ylabel('accuracy')
    plt.title('accuracy comparison')
    plt.legend()

    plt.show()


def draw6():
    cnn_loss, cnn_acc = extract('lstm_log00.txt')
    cnnw_loss, cnnw_acc = extract('lstm_log01.txt')

    plt.subplot(121)
    plt.plot(cnn_loss, label='patience:100')
    plt.plot(cnnw_loss, label='patience:20')
    plt.xlabel('100 steps')
    plt.ylabel('loss')
    plt.title('loss comparison')
    plt.legend()

    plt.subplot(122)
    plt.plot(cnn_acc[:-2], label='patience:100, dev: %.2f, test: %.2f' % (cnn_acc[-2] * 100, cnn_acc[-1] * 100))
    plt.plot(cnnw_acc[:-2], label='patience:20, dev: %.2f, test: %.2f' % (cnnw_acc[-2] * 100, cnnw_acc[-1] * 100))
    plt.xlabel('100 steps')
    plt.ylabel('accuracy')
    plt.title('accuracy comparison')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    draw6()
