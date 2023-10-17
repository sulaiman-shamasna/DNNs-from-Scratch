class EarlyStopping:
    def __init__(self, patience):
        self.cnt = 0
        self.patience = patience

    def update_best_loss(self, current_loss, best_loss):
        if current_loss > best_loss:
            self.cnt += 1
        else:
            best_loss = current_loss
            self.cnt = 0
        return best_loss

    def quit_learning_status(self):
        return self.cnt >= self.patience