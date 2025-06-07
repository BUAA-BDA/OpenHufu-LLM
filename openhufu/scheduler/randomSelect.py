
import time
import random
def client_selection(client_id_set, client_selection_frac, other_info=None):
    # np.random.seed(time.time())
    num_selected = max(int(client_selection_frac * len(client_id_set)), 1)
    selected_clients_set = set(random.sample(client_id_set, num_selected))

    return selected_clients_set