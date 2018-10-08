from preprocessing import preprocessing
from module import batch_generator, decoder_gru,  autoencoder_gru, VAE_gru, Predictor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time 

def main():

    def see_result( i = 234): # inner function / bad habit !
        #result = RNN(x0) # variation=False
        result = RNN(X_long_train[i], X_onehot_train[i], teacher_forcing=True)
        encoding_each = RNN.encode(X_long_train[i])[0][-1].reshape(-1)
        y_predict = predictor(encoding_each)
        print( decipher(demapping, X_onehot_train[i]))
        print( decipher(demapping, result))
        print( decipher(demapping, RNN(X_long_train[i])))
        print("predict: %.3f   actual: %.3f" % (y_predict, y_train[i]) )

    SAVE_PATH = "data/auto_gru_pred"
    continue_training = False
    # load data
    device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu")
    X_onehot_train, X_onehot_test, X_long_train, X_long_test, y_train, y_test, demapping = preprocessing()
    X_onehot_train = [i.to(device) for i in X_onehot_train]
    X_onehot_test = [i.to(device) for i in X_onehot_test]
    X_long_train = [i.to(device) for i in X_long_train]
    X_long_test = [i.to(device) for i in X_long_test]
    y_train = y_train.to(device)
    y_test = y_test.to(device)
    
    # setup, pre-defined hyperparameter
    predictor = Predictor(14)
    RNN =autoencoder_gru(23, 14, device)
    if continue_training:
        print("continue training!")
        RNN.load_state_dict(torch.load(SAVE_PATH))
    RNN.to(device)
    predictor.to(device)
    epoch_n = 60
    batch_size = 20
    count = 0
    optimizer = optim.Adam(RNN.parameters(), lr=0.003)
    optimizerP = optim.Adam(predictor.parameters(), lr=0.003)
    generator = batch_generator(batch_size=batch_size)
    KLD_f = nn.KLDivLoss()
    MSE_f = nn.MSELoss()
    beta = 0.5 # control the loss ratio
    std = 3.846 # the standard deviation for y label
    # training
    print("start training, total epoch:%d" % (epoch_n) )
    trainloss, testloss, test_RMSE, train_RMSE = [], [], [], []
    for epoch in range(epoch_n):
        # optimizer adjusement?
        #if epoch == epoch_n//2:
        #    optimizer = optim.SGD(RNN.parameters(), lr=0.001, momentum=0.9)
        while True:
            x_onehot_batch, x_long_batch, y_train_batch = generator.next_batch(X_onehot_train,X_long_train, y_train)
            loss_train = 0
            loss_Y = torch.tensor([0.]).to(device)
            # calculate reconstruction loss and prediction loss
            for (x_onehot_each, x_long_each, y_train_each) in zip(x_onehot_batch, x_long_batch, y_train_batch):
                x_onehot_each = x_onehot_each.reshape(x_onehot_each.size(0), 1, -1)
                if epoch >= epoch_n * 0.25: # control prediction phase
                    ratio = beta
                if epoch <= epoch_n * 0.15: #control teacher forcing phase
                    # TF training 
                    ratio = 0
                    x_res = RNN(x_long_each, x_onehot_each, teacher_forcing=True) # with TF
                    loss_each = KLD_f(torch.log(x_res), x_onehot_each)
                    loss_train += loss_each
                else:
                    # non-TF training
                    x_res = RNN(x_long_each) # without TF
                    loss_each = KLD_f(torch.log(x_res), x_onehot_each)
                    loss_train += loss_each
                    encoding_each = RNN.encode(x_long_each)[0][-1].reshape(-1)
                    y_pred = predictor(encoding_each)
                    loss_Y += MSE_f(y_pred, y_train_each) * ratio

            #calculate prediction loss
        
            loss = loss_train + loss_Y
            
            # update weight
            optimizer.zero_grad()
            optimizerP.zero_grad()
            loss.backward()
            optimizer.step()
            optimizerP.step()

            # logging
            count += batch_size
            if count % (10*batch_size) == 0:
                # testing set
                acc = accuracy(X_onehot_test, X_long_test, RNN)
                loss_log = 0
                loss_Y_log = 0
                for (x_onehot_each, x_long_each, y_each) in zip(X_onehot_test, X_long_test, y_test):
                    x_onehot_each = x_onehot_each.reshape(x_onehot_each.size(0), 1, -1)
                    x_res = RNN(x_long_each) # without TF
                    loss_each = KLD_f(torch.log(x_res), x_onehot_each)
                    loss_log += loss_each
                    encoding_each = RNN.encode(x_long_each)[0][-1].reshape(-1)
                    y_pred = predictor(encoding_each)
                    loss_Y_log += MSE_f(y_pred, y_each)
                print("time: %s  epoch:%d  train loss X:%.3f  test loss X:%.3f train loss Y:%.3f predicting RMSE loss: %.3f  actual prediction RMSE loss:%.3f accuracy:%d%%   progress:%d%%" % (time.ctime(), epoch, loss_train/len(x_long_batch), loss_log/len(y_test), (loss_Y/len(x_long_batch))**0.5*std ,(loss_Y_log/len(y_test))**0.5 ,(loss_Y_log/len(y_test))**0.5*std  , acc*100, count/len(y_train)/epoch_n*100)  )
                see_result(233)
                print()
                train_RMSE.append( (loss_Y.item()/len(x_long_batch))**0.5*std  )
                test_RMSE.append( (loss_Y_log.item()/len(y_test))**0.5*std)
                trainloss.append(loss_train.item())
                testloss.append(loss_log.item()/len(y_test))
            if generator.new_round:
                break

    cpu = torch.device("cpu")
    RNN.to(cpu)
    torch.save(RNN.state_dict(), SAVE_PATH)
    print(trainloss)
    print(testloss)
    print(train_RMSE)
    print(test_RMSE)

def decipher(demapping,prob):
    ''' translate probability to smiles, only for debug
        ::prob:: expect to be [length, feature_logits]
    '''
    if not torch.is_tensor(prob):
        prob = torch.tensor(prob)
    smiles = ''
    prob = prob.view(-1,prob.shape[-1])
    seq = torch.argmax(prob,dim=1)
    for i in seq:
        smiles += demapping[i.item()]
    return smiles

def accuracy(x_onehot_batch, x_long_batch, RNN):
    correct = 0
    total = 0
    for (x_onehot_each, x_long_each) in zip(x_onehot_batch, x_long_batch):
        x_onehot_each = x_onehot_each.reshape(x_onehot_each.size(0), 1, -1)
        x_pred = RNN(x_long_each)
        x_pred = x_pred.view(-1,x_pred.shape[-1])
        seq_pred = torch.argmax(x_pred, dim=1)
        seq_actual = x_long_each.view(-1)
        total += len(seq_pred)
        correct += (seq_pred == seq_actual).sum().item()
    return correct/total

if __name__ == "__main__":
    main()