import torch
import numpy as np
import tqdm

class SiameseImagePairTrainer():
    """
    Training and Validation done for image pairs 
    
    takes criterion, optimizer and epoch as input and Trains and Validates outputing accuracy and loss at each epoch
    plots the validation and training losses at the end of epochs.
    
    """

    def __init__(self,criterion = None,optimizer = None):

        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, model, train_loader,epoch):

        model.to(device)
        epoch_loss = 0
        epoch_acc = 0

        for imgs1, imgs2, label in tqdm(train_loader):

            #take data to device
            imgs1 = imgs1.to(device)
            imgs2 = imgs2.to(device)
            label = label.to(device)

            # compute output
            output = model(imgs1,imgs2)
            loss = self.criterion(output, label.unsqueeze(1).float())
            acc = binary_acc(output, label.unsqueeze(1).float())

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        print(f'Epoch {epoch+0:03}: | Training Loss: {epoch_loss/len(train_loader):.5f} | Training Acc: {epoch_acc/len(train_loader):.3f}')
        return epoch_loss / len(trainloader)

    def validate(self, model, validationloader,epoch):

        model.to(device)
        valid_loss = 0
        valid_acc = 0

        with torch.no_grad():
            for imgs1, imgs2, label in tqdm(validationloader):

                #take data to device
                imgs1 = imgs1.to(device)
                imgs2 = imgs2.to(device)
                label = label.to(device)

                # compute output
                output = model(imgs1,imgs2)
                loss = self.criterion(output, label.unsqueeze(1).float())
                acc = binary_acc(output, label.unsqueeze(1).float())

                valid_loss += loss.item()
                valid_acc += acc.item()

            print(f'Epoch {epoch+0:03}: | Validation Loss: {valid_loss/len(validationloader):.5f} | Validation Acc: {valid_acc/len(validationloader):.3f}')
            return valid_loss / len(validationloader)

    def fit(self,model,trainloader,validationloader,epochs):

        valid_min_loss = np.Inf
        train_loss_epoch =[]
        test_loss_epoch = []
        for i in range(epochs):

            # switch to train mode
            model.train()
            avg_train_loss = self.train(model,trainloader,i) ###

            model.eval()  # this turns off the dropout lapyer and batch norm
            avg_valid_loss = self.validate(model,validationloader,i) ###

            #add loss of each epoch
            train_loss_epoch.append(avg_train_loss)
            test_loss_epoch.append(avg_valid_loss)

            if avg_valid_loss <= valid_min_loss :
                print("test_loss decreased {} --> {}".format(valid_min_loss,avg_valid_loss))
                torch.save(model.state_dict(),('/content/drive/MyDrive/Assignment_Pixxel/model_best.pt'))
                valid_min_loss = avg_valid_loss

        plt.plot(train_loss_epoch, label='training loss')
        plt.plot(test_loss_epoch, label='test loss')
        plt.title('Loss at the end of each epoch')
        plt.legend()
        plt.savefig('loss_plot.png')

        
class TripletTrainer():
    """
    Training and Validation done for image triplets 
    
    takes criterion, optimizer and epoch as input and Trains and Validates outputing accuracy and loss at each epoch
    plots the validation and training losses at the end of epochs.
    
    """

    def __init__(self,criterion = None,optimizer = None):

        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, model, train_loader,epoch):

        model.to(device)
        epoch_loss = 0.0

        for anchor, pos, neg in tqdm(train_loader):
            #take data to device
            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            # compute output
            anchor_emb, pos_emb, neg_emb = model(anchor,pos,neg)
            loss = self.criterion(anchor_emb, pos_emb, neg_emb)

            # compute gradient and do GD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch+0:03}: | Training Loss: {epoch_loss/len(train_loader):.5f}')
        return epoch_loss / len(trainloader)

    def validate(self, model, validationloader,epoch):

        model.to(device)
        valid_loss = 0.0

        with torch.no_grad():
            for anchor, pos, neg in tqdm(validationloader):

                #take data to device
                anchor = anchor.to(device)
                pos = pos.to(device)
                neg = neg.to(device)

                # compute output
                anchor_emb, pos_emb, neg_emb = model(anchor,pos,neg)
                loss = self.criterion(anchor_emb, pos_emb, neg_emb)

                valid_loss += loss.item()
            
            print(f'Epoch {epoch+0:03}: | Validation Loss: {valid_loss/len(validationloader):.5f}')
            return valid_loss / len(validationloader)

    def fit(self,model,trainloader,validationloader,epochs):

        valid_min_loss = np.Inf
        train_loss_epoch =[]
        test_loss_epoch = []
        for i in range(epochs):

            # switch to train mode
            model.train()
            avg_train_loss = self.train(model,trainloader,i) ###

            model.eval()  # this turns off the dropout lapyer and batch norm
            avg_valid_loss = self.validate(model,validationloader,i) ###

            #add loss of each epoch
            train_loss_epoch.append(avg_train_loss)
            test_loss_epoch.append(avg_valid_loss)

            if avg_valid_loss <= valid_min_loss :
                print("test_loss decreased {} --> {}".format(valid_min_loss,avg_valid_loss))
                torch.save(model.state_dict(),('/content/drive/MyDrive/Assignment_Pixxel/model_best.pt'))
                valid_min_loss = avg_valid_loss

        plt.plot(train_loss_epoch, label='training loss')
        plt.plot(test_loss_epoch, label='test loss')
        plt.title('Loss at the end of each epoch')
        plt.legend()
        plt.savefig('loss_plot.png')
