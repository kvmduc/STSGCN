import torch
import numpy as np
import argparse
import time
import util
from engine import trainer
import os
import os.path as osp

parser = argparse.ArgumentParser()

parser.add_argument('--device',type=str,default='cuda:0',help='')
################ Path to data ################
parser.add_argument('--data',type=str,default='/data/cs.aau.dk/tungkvt/Trafficstream/district3F11T17/FastData/',help='data path')
parser.add_argument('--adjdata',type=str,default='/data/cs.aau.dk/tungkvt/Trafficstream/district3F11T17/graph/',help='adj data path')
################              ################
parser.add_argument('--seq_length',type=int,default=12,help='Input sequence length')
parser.add_argument('--num_for_predict',type=int,default=12,help='Forecast sequence length')
parser.add_argument('--nhid',type=int,default=64,help='Hidden layer dimensions')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=170,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--learning_rate',type=float,default=1e-3,help='learning rate')
parser.add_argument('--epochs',type=int,default=100,help='') # 200
parser.add_argument('--print_every',type=int,default=100,help='Training print')
parser.add_argument('--save',type=str,default='/data/cs.aau.dk/tungkvt/Trafficstream/result/stsgcn/',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--gcn_num',type=int,default=3,help='Number of gcn')
parser.add_argument('--layer_num',type=int,default=4,help='Number of layers')
parser.add_argument('--max_update_factor',type=int,default=1,help='max update factor')
parser.add_argument('--begin_year',type=int,default=2011,help='begin year')
parser.add_argument('--end_year',type=int,default=2011,help='end year')


args = parser.parse_args()

logger = util.init_log()


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed = 0
setup_seed(seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.device

def main():
    for year in range (args.begin_year, args.end_year + 1):

        adj_path = osp.join(args.adjdata, str(year)+"_adj.npz")
        adj_mx = util.load_adj(adj_path)
        vars(args)['num_nodes'] = adj_mx.shape[0]
        adj_mx = util.construct_adj(adj_mx,3).cuda()


        dataloader = util.load_dataset(year, args.data, args.batch_size, args.batch_size, args.batch_size)
        # scaler = dataloader['scaler']

        global_train_steps = dataloader['train_loader'].num_batch

        logger.info(args)

        engine = trainer(args, adj_mx, global_train_steps)

        logger.info("start training year {}".format(year))
        his_loss =[]
        val_time = []
        train_time = []
        for i in range(1,args.epochs+1):
            train_loss = []
            train_mae = []
            train_mape = []
            train_rmse = []
            t1 = time.time()
            dataloader['train_loader'].shuffle()
            for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                trainx = torch.Tensor(x).cuda()
                trainx= trainx
                # print(trainx.shape)             # (b, seq_len, num_node, in_dim)
                trainy = torch.Tensor(y).cuda()
                trainy = trainy
                # print(trainy.shape)             # (b, seq_len, num_node, in_dim)
                metrics = engine.train(trainx, trainy[:,:,:,0])
                train_loss.append(metrics[0])
                train_mae.append(metrics[1])
                train_mape.append(metrics[2])
                train_rmse.append(metrics[3])
                engine.scheduler.step() #
                if iter % args.print_every == 0 :
                    log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                    logger.info(log.format(iter, train_loss[-1], train_mae[-1], train_mape[-1], train_rmse[-1]))
            t2 = time.time()
            train_time.append(t2-t1)

            # val
            valid_loss = []
            valid_mae = []
            valid_mape = []
            valid_rmse = []

            s1 = time.time()
            for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
                testx = torch.Tensor(x).cuda()
                testx = testx
                testy = torch.Tensor(y).cuda()
                testy = testy
                metrics = engine.eval(testx, testy[:,:,:,0])
                valid_loss.append(metrics[0])
                valid_mae.append(metrics[1])
                valid_mape.append(metrics[2])
                valid_rmse.append(metrics[3])
            s2 = time.time()
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            logger.info(log.format(i,(s2-s1)))
            val_time.append(s2-s1)
            mtrain_loss = np.mean(train_loss)
            mtrain_mae = np.mean(train_mae)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            mvalid_loss = np.mean(valid_loss)
            mvalid_mae = np.mean(valid_mae)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)
            his_loss.append(mvalid_loss)

            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            logger.info(log.format(i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mae, mvalid_mape, mvalid_rmse, (t2 - t1)))
            torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
        logger.info("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        logger.info("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))


        # test
        bestid = np.argmin(his_loss)
        engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))

        outputs = []
        realy = torch.Tensor(dataloader['y_test']).cuda()
        realy = realy[:,:,:,0]

        for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).cuda()
            testx = testx
            with torch.no_grad():
                preds = engine.model(testx)
            outputs.append(preds.squeeze())

        yhat = torch.cat(outputs,dim=0)
        yhat = yhat[:realy.size(0),...]
        # print(yhat.shape)   # torch.Size([3567, 12, 170])
        # print(realy.shape)  # torch.Size([3567, 12, 170])
        logger.info("Training finished")
        logger.info("The valid loss on best model is {}".format(str(round(his_loss[bestid],4))))
        logger.info("The epoch of the best model is: {}".format(str(bestid + 1)))

        amae = []
        amape = []
        armse = []
        for i in [2,5,11]:
            # pred = scaler.inverse_transform(yhat[:,i,:])
            pred = yhat[:,i,:]
            real = realy[:,i,:]
            metrics = util.metric(pred,real)
            log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            logger.info(log.format(i+1, metrics[0], metrics[1], metrics[2]))
            amae.append(metrics[0])
            amape.append(metrics[1])
            armse.append(metrics[2])

        log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        logger.info(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
        torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    torch.cuda.empty_cache()
    print("Total time spent: {:.4f}".format(t2-t1))
