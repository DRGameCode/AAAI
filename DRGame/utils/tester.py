import pdb
import logging
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import entropy
from collections import Counter

class Tester(object):
    def __init__(self, args, model, dataloader):
        self.args = args
        self.model = model
        self.history_dic = dataloader.historical_dict
        self.history_csr = dataloader.train_csr
        self.dataloader_test = dataloader.dataloader_test
        self.test_dic = dataloader.test_dic

        self.category = dataloader.category_dic
        self.category_num = dataloader.cate_number

        self.metrics = args.metrics
        self.metrics.append('hited_gt_items')
        self.metrics.append('total_gt_items')

    def judge(self, users, items):
        # items torch.Size([2048, k])

        results = {metric: 0.0 for metric in self.metrics}

        stat = self.stat(items)

        for metric in self.metrics:
            f = Metrics.get_metrics(metric)
            for i in range(len(items)):
                results[metric] += f(items[i], 
                                     test_pos = self.test_dic[users[i]], 
                                     num_test_pos = len(self.test_dic[users[i]]), 
                                     count = stat[i], 
                                     category_num = self.category_num,
                                     model = self.model)
        return results

    def test(self):
        results = {}
        h = self.model.get_embedding()
        count = 0

        for k in self.args.k_list:
            results[k] = {metric: 0.0 for metric in self.metrics}

        for batch in tqdm(self.dataloader_test):

            users = batch[0]
            count += users.shape[0]

            scores = self.model.get_score(h, users)

            users = users.tolist()
            mask = torch.tensor(self.history_csr[users].todense(), device = scores.device).bool()
            scores[mask] = -float('inf')

            _, recommended_items = torch.topk(scores, k = max(self.args.k_list))
            recommended_items = recommended_items.cpu()
            for k in self.args.k_list:

                results_batch = self.judge(users, recommended_items[:, :k])

                for metric in self.metrics:
                    results[k][metric] += results_batch[metric]

        for k in self.args.k_list:
            for metric in self.metrics:
                results[k][metric] = results[k][metric] / count
        self.show_results(results)

    def show_results(self, results):
        for metric in self.metrics:
            if metric == 'hited_gt_items' or metric == 'total_gt_items':
                continue
            for k in self.args.k_list:
                logging.info('For top{}, {} = {}'.format(k, metric, results[k][metric]))
        for k in self.args.k_list:
             logging.info('For top{}, hr* = {}'.format(k, results[k]['hited_gt_items']/results[k]['total_gt_items']))                

    def count_elements(self, lst):
        counter = Counter(lst)
        counts = list(counter.values())
        return counts

    def stat(self, items):
        stat = []
        for item in items:
            category_set = []
            for i in item:
                category_set += self.category[int(i)]
            category_set = set(category_set)
            stat.append(len(category_set))
        return stat



class Metrics(object):

    def __init__(self):
        pass

    @staticmethod
    def get_metrics(metric):

        metrics_map = {
            'recall': Metrics.recall,
            'hit_ratio': Metrics.hr,
            'coverage': Metrics.coverage,
            'precision': Metrics.precision,
            'hited_gt_items': Metrics.hited_gt_items,
            'total_gt_items': Metrics.total_gt_items
        }

        return metrics_map[metric]

    @staticmethod
    def recall(items, **kwargs):

        test_pos = kwargs['test_pos']
        num_test_pos = kwargs['num_test_pos']
        hit_count = np.isin(items, test_pos).sum()

        return hit_count/num_test_pos

    @staticmethod
    def precision(items, **kwargs):
        ## items = TP+FP, test_pos = TP+FN, hit_count = TP
        test_pos = kwargs['test_pos']
        hit_count = np.isin(items, test_pos).sum()

        return hit_count/len(items)


    @staticmethod
    def hr(items, **kwargs):

        test_pos = kwargs['test_pos']
        hit_count = np.isin(items, test_pos).sum()

        if hit_count > 0:
            return 1.0
        else:
            return 0.0

    @staticmethod
    def coverage(items, **kwargs):

        count = kwargs['count']

        return count/kwargs['category_num']


    @staticmethod
    def hited_gt_items(items, **kwargs):

        test_pos = kwargs['test_pos']
        hit_count = np.isin(items, test_pos).sum()
 
        return hit_count

    @staticmethod
    def total_gt_items(items, **kwargs):

        num_test_pos = kwargs['num_test_pos']
        return num_test_pos

