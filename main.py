import math
import random
import torch
import numpy as np
from tqdm import tqdm
from time import time
from prettytable import PrettyTable
import datetime
from utils.parser import parse_args
from utils.data_loader import load_data
from modules.PreFact import PreFact
from utils.evaluate import test
from utils.helper import early_stopping, init_logger
from logging import getLogger
import multiprocessing
import os
import setproctitle
import psutil
import gc

setproctitle.setproctitle("EXP@PreFact")

n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0


class NegativeSampler:
    def __init__(self, n_items, train_user_set, num_neg_sample):
        self.n_items = n_items
        self.train_user_set = train_user_set
        self.num_neg_sample = num_neg_sample

    def __call__(self, user):
        user = int(user)
        each_negs = list()
        neg_item = np.random.randint(low=0, high=self.n_items, size=self.num_neg_sample)
        if len(set(neg_item) & set(self.train_user_set[user])) == 0:
            each_negs += list(neg_item)
        else:
            neg_item = list(set(neg_item) - set(self.train_user_set[user]))
            each_negs += neg_item
            while len(each_negs) < self.num_neg_sample:
                n1 = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if n1 not in self.train_user_set[user]:
                    each_negs += [n1]
        return each_negs


def check_memory_usage():
    memory = psutil.virtual_memory()
    return memory.percent < 80

def get_feed_data(train_cf_pairs, sampler, cores, batch_size=50000):
    def negative_sampling_batch(user_batch):
        user_list = user_batch.cpu().numpy()[:, 0]
        
        max_workers = multiprocessing.cpu_count() // 2
        
        with multiprocessing.get_context("spawn").Pool(processes=max_workers) as pool:
            neg_items = pool.map(sampler, user_list)
        return neg_items
        
    if len(train_cf_pairs) > batch_size:
        logger.info(f"Batch processing: ({len(train_cf_pairs)}), batch size: {batch_size}")
        
        all_neg_items = []
        num_batches = math.ceil(len(train_cf_pairs) / batch_size)
        
        for i in tqdm(range(num_batches), desc="Processing Neg Sampling batches"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(train_cf_pairs))
            batch_pairs = train_cf_pairs[start_idx:end_idx]
            
            if not check_memory_usage():
                logger.info("\nMemory usage is high, forcing garbage collection")
                gc.collect()
                torch.cuda.empty_cache()
            
            batch_neg_items = negative_sampling_batch(batch_pairs)
            all_neg_items.extend(batch_neg_items)
            
            del batch_pairs, batch_neg_items
            gc.collect()
        
        neg_items_tensor = torch.LongTensor(all_neg_items)
        del all_neg_items
    else:
        neg_items_tensor = torch.LongTensor(negative_sampling_batch(train_cf_pairs))
    
    feed_dict = {
        'users': train_cf_pairs[:, 0],
        'pos_items': train_cf_pairs[:, 1],
        'neg_items': neg_items_tensor
    }
    
    return feed_dict


def gamma_schedule(iteration):
    gamma = np.linspace(0, args.gamma, 30000)
    if iteration < 30000:
        return gamma[iteration]
    else:
        return args.gamma


def print_trainable_params(model):
    print("Trainable parameters:")
    total_params = 0
    total_bytes = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            element_size = param.element_size()
            num_bytes = param.numel() * element_size
            total_bytes += num_bytes

            if num_bytes < 1024:
                size_str = f"{num_bytes} B"
            elif num_bytes < 1024 ** 2:
                size_str = f"{num_bytes / 1024:.2f} KB"
            elif num_bytes < 1024 ** 3:
                size_str = f"{num_bytes / (1024 ** 2):.2f} MB"
            else:
                size_str = f"{num_bytes / (1024 ** 3):.2f} GB"

            print(f"- {name}: {param.shape} ({param.numel():,} params, {size_str})")
            total_params += param.numel()

    if total_bytes < 1024:
        total_size_str = f"{total_bytes} B"
    elif total_bytes < 1024 ** 2:
        total_size_str = f"{total_bytes / 1024:.2f} KB"
    elif total_bytes < 1024 ** 3:
        total_size_str = f"{total_bytes / (1024 ** 2):.2f} MB"
    else:
        total_size_str = f"{total_bytes / (1024 ** 3):.2f} GB"

    print(f"Total trainable parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Total memory footprint: {total_size_str}\n")


if __name__ == '__main__':
    """fix the random seed"""
    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device, train_user_set
    args = parse_args()
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

    log_fn = init_logger(args)
    logger = getLogger()

    logger.info("PID: %d", os.getpid())
    logger.info(f"DESC: {args.desc}\n")
    logger.info(f"Args: {args}\n")

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']
    train_user_set = user_dict['train_user_set']

    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    # test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))

    """define model"""
    model = PreFact(n_params, args, graph, mean_mat_list[0]).to(device)
    print_trainable_params(model)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    """create sampler"""
    sampler = NegativeSampler(n_items=n_items, train_user_set=train_user_set, num_neg_sample=args.num_neg_sample)
    cores = multiprocessing.cpu_count()

    cur_best = 0
    stopping_step = 0
    should_stop = False

    # change resume to True if you want to resume training from a saved model
    resume = True
    if resume:
        model_path = os.path.join(args.out_dir, f"model_{args.dataset}.ckpt")
        if os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False).state_dict()
            model.load_state_dict(ckpt, strict=False)
        else:
            logger.info(f"Model file {model_path} not found. Starting from scratch.")

    logger.info("start training ...")
    iter = math.ceil(len(train_cf_pairs) / args.batch_size)
    iteration = 0
    for epoch in range(args.epoch):
        torch.cuda.empty_cache()
        if epoch % 20 == 0:
            # shuffle training data
            index = np.arange(len(train_cf))
            np.random.shuffle(index)
            train_cf_pairs = train_cf_pairs[index]
            logger.info("start preparing feed data ...")
            all_feed_data = get_feed_data(train_cf_pairs, sampler, cores)  # {'user': [n,], 'pos_item': [n,], 'neg_item': [n, n_sample]}

        """training"""
        model.train()
        loss = 0
        train_s_t = time()
        for i in tqdm(range(iter), desc="Epoch %d" % epoch):
            torch.cuda.empty_cache()
            batch = dict()
            batch['users'] = all_feed_data['users'][i*args.batch_size:(i+1)*args.batch_size].to(device)
            batch['pos_items'] = all_feed_data['pos_items'][i*args.batch_size:(i+1)*args.batch_size].to(device)
            batch['neg_items'] = all_feed_data['neg_items'][i*args.batch_size:(i+1)*args.batch_size, :].to(device)

            batch_loss = model(batch, gamma_schedule(iteration))
            
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            loss += batch_loss.item()
            iteration += 1
        train_e_t = time()

        if (epoch + 1) % 10 == 0 or epoch == args.epoch - 1:
            """testing"""
            model.eval()
            test_s_t = time()
            with torch.no_grad():
                ret = test(model, user_dict, n_params, gamma_schedule(iteration))
            test_e_t = time()

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg", "precision", "hit_ratio"]
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss, ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']]
            )
            logger.info(train_res)

            # *********************************************************
            cur_best, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best,
                                                                  stopping_step, expected_order='acc',
                                                                  flag_step=args.flag_step)
            if stopping_step == 0:
                logger.info("###find better!")
            elif should_stop:
                logger.info("###Early stopping is triggered at step: {} log:{}".format(args.flag_step, ret['recall'][0]))
                break

            """save model"""
            if ret['recall'][0] == cur_best and args.save:
                os.makedirs(args.out_dir, exist_ok=True)
                model_path = os.path.join(args.out_dir, f"model_{args.dataset}.ckpt")
                torch.save(model, model_path)
        else:
            logger.info('{}: using time {}, training loss at epoch {}: {}'.format(
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), train_e_t - train_s_t, epoch, loss))

    logger.info('end training at epoch %d, recall@20:%.4f' % (epoch, cur_best))
