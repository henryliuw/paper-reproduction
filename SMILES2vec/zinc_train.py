from zinc_preprocessing import preprocessing
from zinc_module import batch_generator, decoder_gru,  autoencoder_gru, VAE_gru, Predictor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time 
import logging

logger = logging.getLogger('ml')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s    %(levelname)-10s %(message)s')
fh = logging.FileHandler('_ml.log')
ch = logging.StreamHandler()
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

def main():

    def predict_result(i):
        encoding_each = RNN.encode(X_long_test[i])[0][-1].reshape(-1)
        y_predict = predictor(encoding_each)*std + mean
        logger.info( decipher(demapping, X_onehot_test[i]))
        logger.info("predict: %.3f   actual: %.3f" % (y_predict, y_test[i]) )
        logger.info(MSE_f(y_predict, y_test[i]).item())

    def see_result( i = 234): # inner function / bad habit !
        #result = RNN(x0) # variation=False
        result = RNN(X_long_train[i], X_onehot_train[i], teacher_forcing=True)
        encoding_each = RNN.encode(X_long_train[i])[0][-1].reshape(-1)
        y_predict = predictor(encoding_each)
        logger.info( decipher(demapping, X_onehot_train[i]))
        logger.info( decipher(demapping, result))
        logger.info( decipher(demapping, RNN(X_long_train[i])))
        logger.info("predict: %.3f   actual: %.3f" % (y_predict, y_train[i]) )

    SAVE_PATH = "data/auto_gru_ZINC"
    SAVE_PATH2 = "data/predictor_ZINC"

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
    
    # process
    std = y_train.std()
    mean = y_train.mean()
    y_train = (y_train - mean)/std

    # setup, pre-defined hyperparameter
    predictor = Predictor(60)
    RNN =autoencoder_gru(28, 60, device)
    if continue_training:
        print("continue training!")
        RNN.load_state_dict(torch.load(SAVE_PATH))
        predictor.load_state_dict(torch.load(SAVE_PATH2))
    RNN.to(device)
    predictor.to(device)
    
    epoch_n = 60
    batch_size = 1000
    count = 0
    optimizer = optim.Adam(RNN.parameters(), lr=0.008, weight_decay=0.0005)
    optimizerP = optim.Adam(predictor.parameters(), lr=0.003, weight_decay=0.0005)
    generator = batch_generator(batch_size=batch_size)
    KLD_f = nn.KLDivLoss()
    MSE_f = nn.MSELoss()
    beta = 0.5 # control the loss ratio
    # training
    print("start training, total epoch:%d" % (epoch_n) )
    logger.info("start training, total epoch:%d" % (epoch_n) )

    #logging testing
    if continue_training:
        see_result(233)
        for i in range(10):
            predict_result(i)
        print()

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
                if epoch >= epoch_n * 0.25  or continue_training   : # control prediction phase
                    ratio = beta
                if epoch <= epoch_n * 0.15 and (not continue_training) : #control teacher forcing phase
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
            #loss = loss_train
            
            # update weight
            optimizer.zero_grad()
            optimizerP.zero_grad()
            loss.backward()
            optimizer.step()
            optimizerP.step()

            # logging
            count += batch_size
            if count % (5*batch_size) == 0:
                # testing set
                loss_log = 0
                loss_Y_log = 0
                for (x_onehot_each, x_long_each, y_each) in zip(X_onehot_test, X_long_test, y_test):
                    x_onehot_each = x_onehot_each.reshape(x_onehot_each.size(0), 1, -1)
                    x_res = RNN(x_long_each) # without TF
                    loss_each = KLD_f(torch.log(x_res), x_onehot_each)
                    loss_log += loss_each
                    encoding_each = RNN.encode(x_long_each)[0][-1].reshape(-1)
                    y_pred = predictor(encoding_each) * std + mean
                    loss_Y_log += MSE_f(y_pred, y_each)
                acc = accuracy(X_onehot_test, X_long_test, RNN)
                print("time: %s  epoch:%d  train loss X:%.3f  test loss X:%.3f train loss Y:%.3f predicting RMSE loss: %.3f   accuracy:%d%%   progress:%d%%" % (time.ctime(), epoch, loss_train/len(x_long_batch), loss_log/len(y_test), (loss_Y/len(x_long_batch))**0.5*std ,(loss_Y_log/len(y_test))**0.5  , acc*100, count/len(y_train)/epoch_n*100 ) )
                logger.info("epoch:%d  train loss X:%.3f  test loss X:%.3f train loss Y:%.3f predicting RMSE loss: %.3f   accuracy:%d%%   progress:%d%%" % ( epoch, loss_train/len(x_long_batch), loss_log/len(y_test), (loss_Y/len(x_long_batch))**0.5*std ,(loss_Y_log/len(y_test))**0.5  , acc*100, count/len(y_train)/epoch_n*100 ) )
                see_result(233)
                predict_result(25)
                print()
                train_RMSE.append( (loss_Y.item()/len(x_long_batch))**0.5*std.item()  )
                test_RMSE.append( (loss_Y_log.item()/len(y_test))**0.5)
                trainloss.append(loss_train.item())
                testloss.append(loss_log.item()/len(y_test))
            if generator.new_round:
                break

    # save the thing of 

    cpu = torch.device("cpu")
    RNN.to(cpu)
    predictor.to(cpu)
    torch.save(RNN.state_dict(), SAVE_PATH)
    torch.save(predictor.state_dict(), SAVE_PATH2)
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
    count = 0
    for (x_onehot_each, x_long_each) in zip(x_onehot_batch, x_long_batch):
        x_onehot_each = x_onehot_each.reshape(x_onehot_each.size(0), 1, -1)
        x_pred = RNN(x_long_each)
        x_pred = x_pred.view(-1,x_pred.shape[-1])
        seq_pred = torch.argmax(x_pred, dim=1)
        seq_actual = x_long_each.view(-1)
        total += len(seq_pred)
        correct += (seq_pred == seq_actual).sum().item()
        count += 1
        if count == 200: # save computation time
            return correct / total
    return correct / total

if __name__ == "__main__":
    main()
