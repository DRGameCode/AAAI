from tqdm import tqdm
import torch
import logging
import numpy as np
import dgl
from torch.utils.data import DataLoader
from scipy.sparse import coo_matrix
import torch.nn.functional as F
import pickle
import os
from cluster import read_items_embeddings, k_cluster, items_embeds

class Dataloader(object):
    def __init__(self, args, data, device):
        logging.info("loadding data")
        self.args = args
        self.train_path = './datasets/' + data + '/train.txt'
        self.val_path = './datasets/' + data + '/val.txt'
        self.test_path = './datasets/' + data + '/test.txt'

        self.category_path = './datasets/' + data + '/feat2.txt'

        self.train_rating_path = './datasets/' + data + '/train_rating.pickle'
        self.user_item_mat_path = './datasets/' + data + '/user_item_mat.pickle'
        self.user_cate_mat_path = './datasets/' + data + '/user_cate_mat.pickle'
        self.item_cate_mat_path = './datasets/' + data + '/item_cate_mat.pickle'

        self.feat2_path =  './datasets/' + data + "/feat2.txt"
        self.feat1_path = './datasets/' + data + "/feat1.txt"
        self.embeddings_path = './datasets/' + data + "/embeddings.pickle" 


        self.user_number = 0
        self.item_number = 0
        self.cate_number = 0
        self.device = device

        logging.info(' ... reading category information ... ')
        self.category_dic, self.item_cate_mat = self.NEW_read_category(self.category_path)

        logging.info(' ... reading category tensor ... ')
        self.category_tensor = self.NEW_category_tensor(self.category_dic, self.cate_number)

        logging.info(' ... get weight for each sample ... ')
        self.sample_weight = self.NEW_get_sample_weight(self.category_tensor)

        logging.info(' ... reading train data ... ')
        self.train_graph, self.dataloader_train = self.read_train_graph(self.train_path)

        logging.info(' ... reading valid data ... ')
        self.val_graph, self.dataloader_val = self.read_val_graph(self.val_path)

        logging.info(' ... reading test data ... ')
        self.test_dic, self.dataloader_test = self.read_test(self.test_path)




    def get_csr_matrix(self, array):
        users = array[:, 0]
        items = array[:, 1]
        data = np.ones(len(users))
        return coo_matrix((data, (users, items)), shape = (self.user_number, self.item_number), dtype = bool).tocsr()

    def NEW_read_category(self, path):
        dic = {}
        max_category_id = 0
        max_item_id = 0

        with open(path, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip().split(',')
                item = int(line[0])
                category = int(line[1])

                if category > max_category_id:
                    max_category_id = category
                if item > max_item_id:
                    max_item_id = item                    
                
                if item not in dic:
                    dic[item] = [category]
                else:
                    dic[item].append(category)
        self.cate_number = max_category_id + 1
        self.item_number = max_item_id + 1 

        if (os.path.exists(self.item_cate_mat_path)) == 0:

            item_cate_mat = torch.zeros(self.item_number, self.cate_number)  # float32
            for item in dic.keys():
                for cate in dic[item]:
                    item_cate_mat[item, cate] = 1       

            with open(self.item_cate_mat_path, 'wb') as f:
                pickle.dump(item_cate_mat, f)   
        else:
            with open(self.item_cate_mat_path, 'rb') as f:
                item_cate_mat = pickle.load(f)                          

        return dic, item_cate_mat


    def NEW_category_tensor(self, dic, num):
        category_list = list(dic.values())
        one_hot_list = []
        for sublist in category_list:
            one_hot_sublist = F.one_hot(torch.tensor(sublist), num_classes=num).sum(dim=0)
            one_hot_list.append(one_hot_sublist)
        category_tensor = torch.stack(one_hot_list)   
        return category_tensor
    
    def NEW_get_sample_weight(self, category_tensor):
        category_count = category_tensor.sum(dim=0).float()

        print(category_count.shape)

        weight = 1 / category_count

        weight = weight / weight.sum() * self.cate_number
        weight = weight.unsqueeze(-1)
        return torch.mm(category_tensor.float(), weight).squeeze(1)


    def NEW_rating(self, data_tensor):
        item_time_dic = {}
        item_list = data_tensor[:, 1].numpy().tolist()
        time_list = data_tensor[:, 2].numpy().tolist()
        length = len(item_list)
        for i in tqdm(range(length)):
            item = item_list[i]
            time = time_list[i]
            if item not in item_time_dic:
                item_time_dic[item] = [time]
            else:
                item_time_dic[item].append(time)

        dic_keys = list(item_time_dic.keys())
        for key in dic_keys:
            times_sorted = sorted(item_time_dic[key])
            item_time_dic[key] = times_sorted
    
        rating_list = []
        for i in tqdm(range(length)):
            item = item_list[i]
            time = time_list[i]
            rating = (item_time_dic[item].index(time) + item_time_dic[item].count(time))/float(len(item_time_dic[item]))

            rating_list.append(rating)
        rating_tensor = torch.Tensor(rating_list)
        with open(self.train_rating_path, 'wb') as f:
            pickle.dump(rating_tensor, f)
        return rating_tensor



    def GIPR(self, ui_mat, ic_mat):
        if (os.path.exists(self.user_cate_mat_path)) == 0:

            cate_stat = torch.sum(ic_mat, dim = 0)
            user_cate_mat = torch.mm(ui_mat.type(torch.bool).type(torch.float32), ic_mat)
            for i in tqdm(range(user_cate_mat.shape[0])):
                u_emb = user_cate_mat[i]

                if torch.sum(u_emb, dim = 0) == 0:
                    continue

                u_emb = u_emb / torch.sum(u_emb, dim = 0)
                c_emb = u_emb.type(torch.bool).type(torch.float32) * cate_stat
                c_emb = c_emb/torch.sum(c_emb, dim = 0)

                temp_emb = []
                for j in range(u_emb.shape[0]):
                    if u_emb[j] == 0:
                        temp_emb.append(0)
                    else:
                        temp_emb.append(u_emb[j].item() / c_emb[j].item())
                user_cate_mat[i] = torch.Tensor(temp_emb)


            with open(self.user_cate_mat_path, 'wb') as f:
                pickle.dump(user_cate_mat, f)   
        else:
            with open(self.user_cate_mat_path, 'rb') as f:
                user_cate_mat = pickle.load(f)            

        self.user_cate_mat = user_cate_mat
        return ui_mat * torch.mm(user_cate_mat, ic_mat.t())



    def RNR(self, ui_score_mat):
        graph_path = self.feat2_path
        feat_path = self.feat1_path
        embeddings_path = self.embeddings_path

        f = open(feat_path, 'w', encoding='utf-8')
        for i in range(self.item_number + self.cate_number):
            f.write(str(i) + "," + str(i) + "\n")
        f.close()

        if (os.path.exists(embeddings_path)) == 0:
            logging.info('items embeddings does not exist!')
            embeddings = items_embeds(graph_path, feat_path, embeddings_path, self.args, self.item_number)
            logging.info('save items embeddings!')
        else:
            logging.info('load items embeddings...')

            embeddings = read_items_embeddings(embeddings_path)
        logging.info('finish items embeddings...')

        item_embeddings = embeddings[ : self.item_number]
        cate_embeddings = embeddings[self.item_number : ]

        item_embeddings_plus = torch.mm(self.item_cate_mat, cate_embeddings.cpu())
        item_embeddings = torch.cat([item_embeddings.cpu(), item_embeddings_plus], dim=1)
        user_embeddings = torch.mm(self.user_cate_mat, cate_embeddings.cpu())

        Beta1 = self.args.p1
        Beta2 = self.args.p2

        print("### user_number: ", self.user_number)
        print("### item_number: ", self.item_number)
        print("### cate_number: ", self.cate_number)

        item_ids, item_subjects = k_cluster(item_embeddings, int(self.item_number * Beta1))
        user_ids, user_subjects = k_cluster(user_embeddings, int(self.user_number * Beta2))
        item_subjects = torch.Tensor(item_subjects)
        user_subjects = torch.Tensor(user_subjects)


        new_u2i_mat = torch.zeros(ui_score_mat.shape)
        logging.info('... user nodes selection ...')
        for u in tqdm(range(ui_score_mat.shape[0])):
            u2i_vec = ui_score_mat[u]
            s_ir_dic = {}
            for i in range(u2i_vec.shape[0]):
                if u2i_vec[i].item() > 0:
                    s = item_subjects[i].item()
                    if s not in s_ir_dic:
                        s_ir_dic[s] = [i, u2i_vec[i].item()]
                    else:
                        if u2i_vec[i].item() > s_ir_dic[s][1]:
                            s_ir_dic[s] = [i, u2i_vec[i].item()]
            for s in s_ir_dic:
                new_u2i_mat[u][s_ir_dic[s][0]] = s_ir_dic[s][1]

        new_i2u_mat = torch.zeros(ui_score_mat.t().shape)
        logging.info('... item nodes selection ...')
        for i in tqdm(range(ui_score_mat.shape[1])):
            i2u_vec = ui_score_mat.t()[i]
            s_ur_dic = {}
            for u in range(i2u_vec.shape[0]):
                if i2u_vec[u].item() > 0:
                    s = user_subjects[u].item()
                    if s not in s_ur_dic:
                        s_ur_dic[s] = [u, i2u_vec[u].item()]
                    else:
                        if i2u_vec[u].item() > s_ur_dic[s][1]:
                            s_ur_dic[s] = [u, i2u_vec[u].item()]
            for s in s_ur_dic:
                new_i2u_mat[i][s_ur_dic[s][0]] = s_ur_dic[s][1]
        
        return new_u2i_mat, new_i2u_mat


    def read_train_graph(self, path):
        self.historical_dict = {}
        train_data = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip().split(',')
                user = int(line[0])
                item = int(line[1])
                time = int(line[2])
                train_data.append([user, item, time])

                if user in self.historical_dict:
                    self.historical_dict[user].add(item)
                else:
                    self.historical_dict[user] = set([item])

        train_data = torch.tensor(train_data)
        self.user_number = max(self.user_number, train_data[:, 0].max() + 1)
        if self.item_number != (train_data[:, 1].max() + 1):
            print(" !!! ERROR: item_num !!! ")
        # self.item_number = max(self.item_number, train_data[:, 1].max() + 1)
        
        if (os.path.exists(self.user_item_mat_path)) == 0:
            if (os.path.exists(self.train_rating_path)) == 0:
                r = self.NEW_rating(train_data)
            else:
                with open(self.train_rating_path, 'rb') as f:
                    r = pickle.load(f)

            user_item_mat = torch.zeros(self.user_number, self.item_number)  # float32
            for i in tqdm(range(train_data.shape[0])):
                user = train_data[i, 0]
                item = train_data[i, 1]
                s = r[i]
                user_item_mat[user, item] = s
            with open(self.user_item_mat_path, 'wb') as f:
                pickle.dump(user_item_mat, f)   

        else:
            with open(self.user_item_mat_path, 'rb') as f:
                user_item_mat = pickle.load(f)   


        gipr_mat = self.GIPR(user_item_mat, self.item_cate_mat)
              
        u2i_mat, i2u_mat = self.RNR(gipr_mat)


        self.train_csr = self.get_csr_matrix(train_data)
        graph_data = {
            ('user', 'play', 'item'): (train_data[:, 0].long(), train_data[:, 1].long()),
            ('item', 'played by', 'user'): (train_data[:, 1].long(), train_data[:, 0].long())
        }
        graph = dgl.heterograph(graph_data)
        
        dataset = torch.utils.data.TensorDataset(train_data)
        dataloader = DataLoader(dataset, batch_size = self.args.batch_size, shuffle = True, num_workers = 4)

        u2i_mask = []
        for k in tqdm(range(train_data.shape[0])):
            u = train_data[k, 0].item()
            i = train_data[k, 1].item()
            if u2i_mat[u, i] > 0:
                u2i_mask.append(1)
            else:
                u2i_mask.append(0)

        i2u_mask = []
        for k in tqdm(range(train_data.shape[0])):
            u = train_data[k, 0].item()
            i = train_data[k, 1].item()
            if i2u_mat[i, u] > 0:
                i2u_mask.append(1)
            else:
                i2u_mask.append(0)
        u2i_mask = torch.Tensor(u2i_mask)
        i2u_mask = torch.Tensor(i2u_mask)

        graph.edges['play'].data['mask'] = i2u_mask
        graph.edges['played by'].data['mask'] = u2i_mask


        return graph.to(self.device), dataloader


    def read_val_graph(self, path):
        val_data = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip().split(',')
                user = int(line[0])
                item = int(line[1])
                # time = int(line[2])
                val_data.append([user, item])

        val_data = torch.tensor(val_data)

        graph_data = {
            ('user', 'play', 'item'): (val_data[:, 0].long(), val_data[:, 1].long()),
            ('item', 'played by', 'user'): (val_data[:, 1].long(), val_data[:, 0].long())
        }
        number_nodes_dict = {'user': self.user_number, 'item': self.item_number}
        graph = dgl.heterograph(graph_data, num_nodes_dict = number_nodes_dict)

        dataset = torch.utils.data.TensorDataset(val_data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.args.batch_size, shuffle = True)

        return graph.to(self.device), dataloader
    
    def read_test(self, path):
        dic_test = {}
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(',')
                user = int(line[0])
                item = int(line[1])
                if user in dic_test:
                    dic_test[user].append(item)
                else:
                    dic_test[user] = [item]
        
        dataset = torch.utils.data.TensorDataset(torch.tensor(list(dic_test.keys()), dtype = torch.long, device = self.device))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.args.batch_size, shuffle = False)
        return dic_test, dataloader
    
