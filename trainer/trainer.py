import numpy as np
import logging
import torch
from torch.utils.data import DataLoader

from utils import utils, signal_utils, file_utils
import trainer.loss as Loss

# A logger for this file
log = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, datasets, model_cfg):
        # datasets: dict of keys = {"train", "validation", "test"}
        self.model = model.cuda() if torch.cuda.is_available() else model
        log.info(model.__class__)
        self.optimizer = self.create_optimizer(model_cfg.train.optimizer)

        self.datasets = datasets
        self.model_cfg = model_cfg
        
    def _get_dataloader(self, key, dataloader_cfg):
        if self.datasets.get(key) is None:
            raise ValueError(f"Trainer: missing datasets[{key}].")
        return DataLoader(dataset=self.datasets[key],
            batch_size=dataloader_cfg.batch_size,
            shuffle=dataloader_cfg.shuffle,
            num_workers=dataloader_cfg.num_workers
        )
        

    def create_optimizer(self, optimizer_cfg):
        trainable_params = [p for n, p in self.model.named_parameters()]
        return torch.optim.Adam(trainable_params, lr = optimizer_cfg.lr)

    def create_scheduler(self):
        pass

    def compute_loss(self, x, z, m, xhat, loss_cfg, returnSub=False):
        return Loss.get_loss(x=x, z=z, m=m, xhat=xhat, loss_cfg=loss_cfg, returnSub=returnSub)

    #@utils.timeit
    def training_step(self, dataloader):
        loss_step, n = 0.0, 0
        self.model.train()# Optional when not using Model Specific layer
        
        #sub_loss_arr = []
        for x, z, m in dataloader:
            n += 1
            if torch.cuda.is_available():
                x, z, m = x.cuda(), z.cuda(), m.cuda()
            self.optimizer.zero_grad()
            xhat = self.model(x=x, z=z, m=m)
            #loss, sub = self.compute_loss(x=x, z=z, m=m, xhat=xhat, loss_cfg=self.model_cfg.loss, returnSub=True) #(batch_size,)
            loss = self.compute_loss(x=x, z=z, m=m, xhat=xhat, loss_cfg=self.model_cfg.loss) #(batch_size,)
            loss = loss.mean()

            #sub = [sb.mean().item() for sb in sub]
            #sub_loss_arr.append(sub)
            
            loss.backward()
            self.optimizer.step()
            
            loss_step += loss.item()
        #print(np.mean(np.array(sub_loss_arr), axis=0))

        loss_step /= n
        return loss_step

    def evaluate_step(self, dataloader, fig_path=None):
        '''
            Returns:
                loss: float, (1,), average loss
                xhat: numpy array, (n, window_size), model output
        '''
        loss_step, n = 0.0, 0
        xhat_z_arr = []
        
        self.model.eval()# Optional when not using Model Specific layer
        for x, z, m in dataloader:
            n += len(x)
            if torch.cuda.is_available():
                x, z, m = x.cuda(), z.cuda(), m.cuda()
            with torch.no_grad():
                xhat = self.model(x=x, z=z, m=m)
                loss = self.compute_loss(x=x, z=z, m=m, xhat=xhat, loss_cfg=self.model_cfg.loss) #(batch_size,)
                loss = loss.mean()
                
                loss_step += loss.sum().item() #mean later over all loss can reduce error in float-point
                xhat_z_arr.extend(
                    np.hstack((xhat.detach().cpu().numpy()[:, np.newaxis, :],
                            z.detach().cpu().numpy()[:, np.newaxis, :])))
        loss_step /= n
        xhat_z_arr = np.array(xhat_z_arr)
        
        cfg = self.model_cfg.evaluate
        if cfg.do_plot:
            self._plot(x=x[0].detach().cpu().numpy(),
                xhat=xhat_z_arr[:, 0, :], # (n, ws)
                m=m[0].detach().cpu().numpy(), 
                path=fig_path, cfg=cfg.plot)
        return loss_step, xhat_z_arr
    
    def _plot(self, x, xhat, m, path, cfg):
        '''
            x: numpy array, (window_size,)
            xhat: numpy array, (n, window_size)
            m: numpy array, ()
        '''
        if path is not None:
            path = utils.get_hydra_output_path(path)
        signal_utils.plot_FFT(x=x, xhat=xhat, m=m, save_path=path, cfg=cfg)

    def train(self):
        cfg = self.model_cfg.train

        # Get dataloaders from datasets
        train_dataloader = self._get_dataloader("train", cfg.dataloader)
        if cfg.do_evaluate:
            eval_dataloader = self._get_dataloader("validation", self.model_cfg.evaluate.dataloader)

        # Load model
        if cfg.checkpoint.do_load:
            self._load_checkpoint(cfg.checkpoint.load_path)
        
        # Per epoch: train, [eval], [log_metric], [save_checkpoint]
        min_loss = np.inf
        train_loss_arr, eval_output_arr = [], []
        for e in range(cfg.n_epoch):
            # Training step
            loss = self.training_step(train_dataloader)
            train_loss_arr.append(loss)
            msg = f'🍺 Epoch {e+1}/{cfg.n_epoch} \t Training Loss: {loss}'
            
            # Validation step
            if cfg.do_evaluate:
                eval_loss, outputs = self.evaluate_step(eval_dataloader, fig_path=f'./evalEpoch{e}_fft.pdf')
                eval_output_arr.append({'xhat_z_arr': outputs, 'loss': eval_loss})
                msg += f'\t Validation Loss: {eval_loss}'
                if cfg.anchor_eval_loss: loss = eval_loss

            log.info(msg)
            
            # Save checkpoint
            if cfg.checkpoint.do_save and min_loss > loss: # Save if loss goes down
                self._save_checkpoint(loss_old=min_loss, loss_new=loss, path=cfg.checkpoint.save_path)
                min_loss = loss
        
        # save eval xhat, z, and loss
        if cfg.do_evaluate and self.model_cfg.evaluate.output.do_save:
            path = utils.get_hydra_output_path(self.model_cfg.evaluate.output.path)
            file_utils.dumpPkl(eval_output_arr, path)

    def predict(self):
        cfg = self.model_cfg.predict
        dataloader = self._get_dataloader("test", cfg.dataloader)
        # Load model
        if cfg.checkpoint.do_load:
            self._load_checkpoint(cfg.checkpoint)
        
        # Evaluate
        loss, outputs = self.evaluate_step(dataloader, fig_path='./predict_fft.pdf')
        log.info("🍺 Prediction loss = {}, outputs = {}".format(loss, outputs.shape))
        if cfg.output.do_save:
            path = utils.get_hydra_output_path(cfg.output.path)
            file_utils.dumpPkl(outputs, path)

    def _load_checkpoint(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location = torch.device('cpu')))
        #log.info(next(self.model.parameters()).device)
        log.info("Loaded {}".format(model_path))

    def _save_checkpoint(self, loss_old, loss_new, path):
        # Saving State Dict
        log.info(f'Loss Decreased({loss_old:.6f}--->{loss_new:.6f}) \t Saving the Model in {path}')
        torch.save(self.model.state_dict(), path)

