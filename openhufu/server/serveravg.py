import time
from openhufu.server.serverbase import Server


class serverAvg(Server):
    def __init__(self, cfg, times):
        super().__init__(cfg, times)

        # select slow clients
        # self.set_slow_clients()
        # self.set_clients(clientAVG)
        # TODO:在初始化函数里加载模型 暂时不管慢发送的客户端 在deploy里设置客户端 看一下寇学长怎么弄的 把客户端放在了哪里
        # print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        # print("Finished creating server and clients.")

        # self.load_model()
        # self.Budget = []


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            # send_model的同时触发train
            # self.comm_manager.send_model()
            # client向id的映射存在 comm_manager里 server里只存id
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            # 这一步要触发 由comm_manager来协调？ 从client函数里由client的send_message函数主动触发

            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()