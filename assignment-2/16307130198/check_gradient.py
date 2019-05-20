import numpy as np


epsilon = 1e-8


def softmax_layer(input_data):
    #input_data = input_data - np.max(input_data, axis=1).reshape(-1,1)
    numerator = np.exp(input_data)
    denominator = np.expand_dims(np.sum(numerator, axis=1), axis=1)
    softmax_output = numerator / denominator
    return softmax_output

def cross_entropy_loss(predictions, labels):
    epsilon_ = 1e-8
    float_labels = np.array(labels).astype(np.float32)
    cross_entropy_loss_val = (-1.0) * float_labels * np.log(predictions + epsilon_)
    cross_entropy_loss_val = np.mean(np.sum(cross_entropy_loss_val, axis=1))
    return cross_entropy_loss_val

def forward(X,Y,W):
    fc_output = np.dot(X, np.transpose(W))
    softmax_output = softmax_layer(fc_output)
    return cross_entropy_loss(softmax_output, Y)


def numerical_calculate_gradient(X,Y,W):
    example_num = X.shape[0]
    a1, a2 = W.shape
    numerical_gradient = np.zeros((a1, a2))

    former_loss = forward(X,Y,W)

    for i in range(a1):
        for j in range(a2):
            W[i][j] += epsilon
            
            new_loss = forward(X,Y,W)
            numerical_gradient[i][j] = (new_loss - former_loss) / epsilon
            W[i][j] -= epsilon 
    
    return numerical_gradient


def formula_calculate_gradient(X,Y,W):
    example_num = X.shape[0]
    predictions = softmax_layer(np.dot(X, np.transpose(W)))

    gradient = np.dot(np.transpose(X), predictions - Y)/example_num
    return np.transpose(gradient)

if __name__ == "__main__":
    np.random.seed(0)
    # generate inputs data
    X = np.random.rand(5,10)
    Y = np.eye(4)[np.random.randint(0,4,5)]
    W = np.random.rand(4, 10)

    numerical_gradient = numerical_calculate_gradient(X,Y,W)
    formula_gradient = formula_calculate_gradient(X,Y,W)
    
    dif = np.abs(numerical_gradient-formula_gradient)
    print(dif)
    print(np.mean(dif))
