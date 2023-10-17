import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import multilabel_confusion_matrix
import warnings
warnings.filterwarnings('ignore')
epsilon = 1e-15
from early_stopping import EarlyStopping
from datetime import datetime

current_time = str(datetime.now())

class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._current_epoch = 0
        self._early_stopping_patience = early_stopping_patience
        self.setup_cuda()
        self.batch_size = 64
        self.best_loss = float('inf')
        self.best_F1 = 0

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def setup_cuda(self, cuda_device_id=0):
        if t.cuda.is_available():
            t.backends.cudnn.fastest = True
            t.cuda.set_device(cuda_device_id)
            self.device = t.device('cuda')
            t.cuda.manual_seed_all(42)
            t.manual_seed(42)
        else:
            self.device = t.device('cpu')

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        # Forward pass.
        self._optim.zero_grad()
        with t.set_grad_enabled(True):
            output = self._model(x)
            label = y.float()
            loss = self._crit(output, label)

            # Backward and optimize
            loss.backward()
            self._optim.step()
        return loss.item(), output
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss

    def val_test_step(self, x, y):
        output = self._model(x)
        label = y.float()
        loss = self._crit(output, label)

        return loss.item(), output
        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions

    def train_epoch(self):
        self._model.train()
        total_train_loss = 0
        batch_count = 0
        batch_count_print = 0
        predictions_arr = t.from_numpy(np.zeros((len(self._train_dl) * self.batch_size, 2)))
        labels_arr = t.from_numpy(np.zeros_like(predictions_arr))
        for idx, (image, label) in enumerate(self._train_dl):
            image = image.to(self.device)
            label = label.to(self.device)
            train_loss, output = self.train_step(image, label)
            # print(f'set of batchs number: {idx}, has loss: {train_loss}')
            total_train_loss += train_loss

            predicted_labels = (output > 0.5).float()

            for i, batch in enumerate(predicted_labels):
                predictions_arr[idx * self.batch_size + i] = batch
            for i, batch in enumerate(label):
                labels_arr[idx * self.batch_size + i] = batch

            batch_count += 1
            batch_count_print += 1
            # Prints loss statistics after number of steps specified.
            # if (idx + 1) % 10 == 0:
            #     print('Epoch {:02} | Batch 10 | Train loss: {:.3f}'.
            #           format(self._current_epoch, total_train_loss / batch_count_print))
            #     batch_count_print = 0

        crack_confusion_matrix, inactive_confusion_matrix = multilabel_confusion_matrix(labels_arr.cpu(),
                                                                          predictions_arr.cpu())
        TN = crack_confusion_matrix[0, 0]
        FP = crack_confusion_matrix[0, 1]
        FN = crack_confusion_matrix[1, 0]
        TP = crack_confusion_matrix[1, 1]
        accuracy_crack = (TP + TN) / (TP + TN + FP + FN + epsilon)
        F1_crack = 2 * TP / (2 * TP + FN + FP + epsilon)

        TN_inactive = inactive_confusion_matrix[0, 0]
        FP_inactive = inactive_confusion_matrix[0, 1]
        FN_inactive = inactive_confusion_matrix[1, 0]
        TP_inactive = inactive_confusion_matrix[1, 1]
        accuracy_inactive = (TP_inactive + TN_inactive) / (
                TP_inactive + TN_inactive + FP_inactive + FN_inactive + epsilon)
        F1_inactive = 2 * TP_inactive / (2 * TP_inactive + FN_inactive + FP_inactive + epsilon)

        epoch_accuracy = (accuracy_crack + accuracy_inactive) / 2
        epoch_f1_score = (F1_crack + F1_inactive) / 2
        # print(f'the number of batches is: {batch_count}')
        epoch_loss = total_train_loss / batch_count
        # print("------------------------------------------")
        print(f'\t epoch {self._current_epoch} Train Loss: {epoch_loss:.3f} | Train Acc: {epoch_accuracy * 100:.2f}% | Train F1: {epoch_f1_score:.3f}')
        # print("------------------------------------------")

        return epoch_loss
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it

    def val_test(self):
        self._model.eval()
        total_validation_loss = 0
        batch_count = 0
        batch_count_print = 0
        with t.no_grad():

            predictions_arr = t.from_numpy(np.zeros((len(self._val_test_dl) * self.batch_size, 2)))
            labels_arr = t.from_numpy(np.zeros_like(predictions_arr))

            for idx, (image, label) in enumerate(self._val_test_dl):
                image = image.to(self.device)
                label = label.to(self.device)
                val_loss, output = self.val_test_step(image, label)
                total_validation_loss += val_loss
                predicted_labels = (output > 0.5).float()

                for i, batch in enumerate(predicted_labels):
                    predictions_arr[idx * self.batch_size + i] = batch
                for i, batch in enumerate(label):
                    labels_arr[idx * self.batch_size + i] = batch

                batch_count += 1
                batch_count_print += 1
                # Prints loss statistics after number of steps specified.
                # if (idx + 1) % 10 == 0:
                #     print('Epoch {:02} | Batch 10 | Validation loss: {:.3f}'.
                #           format(self._current_epoch, total_validation_loss / batch_count_print))
                #     batch_count_print = 0

            crack_confusion_matrix, inactive_confusion_matrix = multilabel_confusion_matrix(labels_arr.cpu(),
                                                                              predictions_arr.cpu())

            TN = crack_confusion_matrix[0, 0]
            FP = crack_confusion_matrix[0, 1]
            FN = crack_confusion_matrix[1, 0]
            TP = crack_confusion_matrix[1, 1]
            accuracy_crack = (TP + TN) / (TP + TN + FP + FN + epsilon)
            F1_crack = 2 * TP / (2 * TP + FN + FP + epsilon)

            TN_inactive = inactive_confusion_matrix[0, 0]
            FP_inactive = inactive_confusion_matrix[0, 1]
            FN_inactive = inactive_confusion_matrix[1, 0]
            TP_inactive = inactive_confusion_matrix[1, 1]
            accuracy_inactive = (TP_inactive + TN_inactive) / (
                        TP_inactive + TN_inactive + FP_inactive + FN_inactive + epsilon)
            F1_inactive = 2 * TP_inactive / (2 * TP_inactive + FN_inactive + FP_inactive + epsilon)

            epoch_accuracy = (accuracy_crack + accuracy_inactive) / 2
            epoch_f1_score = (F1_crack + F1_inactive) / 2
            epoch_loss = total_validation_loss / batch_count
            # print(f'the number of batches is: {batch_count}')
            # print("------------------------------------------")
            print(f'\t -----------------------------------------------------------------> epoch {self._current_epoch} Val. Loss: {epoch_loss:.3f} |  Val. Acc: {epoch_accuracy * 100:.2f}% |  Val. F1: {epoch_f1_score:.3f}')
            # print("------------------------------------------")
            # print(f'epoch - accuracy: {epoch_accuracy} - f1-score: {epoch_f1_score} - epoch - loss: {epoch_loss}')
            return epoch_loss, epoch_f1_score, epoch_accuracy

        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore. 
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0, "Please check the early stopping patience period and " \
                                                                "the number of epochs "
        train_losses = []
        validation_losses = []
        # create a list for the train and validation losses, and create a counter for the epoch
        # TODO

        #define early-stopping
        early_stopping_checker = EarlyStopping(patience=self._early_stopping_patience)

        best_valid_loss = float('inf')
        while True:
            self._current_epoch += 1
            if self._current_epoch > epochs:
                break
            train_loss = self.train_epoch()
            validation_loss, validation_F1_score, _ = self.val_test()
            train_losses.append(train_loss)
            validation_losses.append(validation_loss)

            # save the model
            # if validation_loss < self.best_loss:
            #     self.best_loss = validation_loss
            #     self.save_checkpoint(self._current_epoch)

            if validation_F1_score > self.best_F1:
                self.best_F1 = validation_F1_score
                self.save_checkpoint(self._current_epoch)



            # Early Stopping
            best_valid_loss = early_stopping_checker.update_best_loss(current_loss=validation_loss, best_loss=best_valid_loss)
            stopping_status = early_stopping_checker.quit_learning_status()
            if stopping_status:
                print("Early stopping has been activated")
                print(f'Stopping at epoch number: {self._current_epoch}')
                break
        # print(f'Avg. train losses: {np.mean(train_losses)}, Avg. validation losses: {np.mean(validation_losses)}')
        return train_losses, validation_losses

        # train for a epoch and then calculate the loss and metrics on the validation set
        # append the losses to the respective lists
        # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
        # check whether early stopping should be performed using the early stopping criterion and stop if so
        # return the losses for both training and validation
