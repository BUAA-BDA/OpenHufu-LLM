
import torch
import os
from torch.nn.functional import normalize
from  openhufu.private.utlis.util import get_logger

logger  = get_logger('AGG_FUNCTION')
def FedAvg(selected_clients_set,id2params, id2weight, epoch):
    # 归一化聚合权重
    # weights_array = normalize(
    #     torch.tensor([id2weight[client_id] for client_id in selected_clients_set],
    #                  dtype=torch.float32),
    #     p=1, dim=0)

    weight_sum = sum(value for value in id2weight.values())
    weighted_single_weights = None
    for client_id in selected_clients_set:
        # single_output_dir = os.path.join(output_dir, str(epoch), "local_output_{}".format(client_id),
        #                                  "pytorch_model.bin")
        # single_weights = torch.load(single_output_dir)
        # why it is a tuple?
        single_weights = id2params[client_id][0]
        print(f'fedavg:{type(single_weights)}')
        # print(single_weights)
        if weighted_single_weights == None:
             weighted_single_weights = {key: single_weights[key] * (id2weight[client_id]) for key in
                                       single_weights.keys()}
        else:
            weighted_single_weights = {key: weighted_single_weights[key] + single_weights[key] * (id2weight[client_id])
                                       for key in
                                       single_weights.keys()}
    return weighted_single_weights