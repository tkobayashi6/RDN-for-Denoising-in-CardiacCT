
import os
import os.path
import torch
import pandas as pd
from os.path import join
from datetime import datetime, timedelta, timezone


class SaveData():
    def __init__(self, args):
        self.args = args
        self.timezone = timezone(timedelta(hours=+9), 'JST')
        self.train_dir = datetime.now(self.timezone).strftime("%Y%m%d_%H%M")
        self.checkpoint = os.path.join(
            args.checkpoint,
            'RDB{}C{}G{}_{}_b{}p{}'.format(
                args.D,
                args.C,
                args.G,
                args.activation,
                args.batch_size,
                args.patch_size
            ),
            self.train_dir
        )
        self.save_model_path = os.path.join(self.checkpoint, 'model')

    def create_checkpoint_dir(self):
        os.makedirs(self.checkpoint)
        os.makedirs(self.save_model_path)

    def write_args(self):
        with open(os.path.join(self.checkpoint, 'args.txt'), 'w') as f:
            for k, v in sorted(self.args.__dict__.items()):
                f.write(f'{k}={v}\n')

    def write_loss(self, loss_results):
        save_path = os.path.join(self.checkpoint, 'results_loss.csv')
        pd.DataFrame(loss_results).to_csv(save_path, index=False)

    def save_model(self, model, model_name):
        torch.save(model.state_dict(), join(self.save_model_path, model_name))

    def keep_records(self, training_time, bestscore_epoch_train, bestscore_epoch_val):
        records = 'Workdir: {}\n'.format(self.checkpoint)
        records += 'Training time:{:.2f}h\n'.format(training_time/3600)
        records += 'best_model_train.pt: epoch={}\n'.format(bestscore_epoch_train)
        records += 'best_model_val.pt: epoch={}'.format(bestscore_epoch_val)
        with open(os.path.join(self.checkpoint, 'records.txt'), mode='w') as f:
            f.write(records)
