import time
from options import args_parser
from my_utils import *
from model_pool_resnet import *
from model_pool_vgg import *
from support_tools import *
from model_prepare import *
from update_new import *
import copy
import warnings
warnings.filterwarnings("ignore", category=Warning)
torch.manual_seed(10)
np.random.seed(10)

if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()

    exp_details(args)
    device = args.device
    print('Device: ', device)


    train_dataset, test_dataset, _, user_groups = get_dataset(args)
    model_dict = output_dict

    if args.core_model == 'resnet18':
        core_model = ResNet18(args.num_classes)
    core_model = core_model.to(device)

    test_acc_list = []

    user_local_model = {}
    user_local_model_name = {}
    

    for idx in range(args.num_users):
    
        local_model = model_dict[idx % len(model_dict)].model
        local_model.to(device)
        local_model = LocalUpdate(args=args, dataset=train_dataset, idxs = user_groups[idx], large_model=copy.deepcopy(local_model), small_model=copy.deepcopy(core_model))
        user_local_model[idx] = local_model
        user_local_model_name[idx] = model_dict[idx % len(model_dict)].name

    for epoch in tqdm(range(args.communication_round)):
        print(f'\n | Communication Round : {epoch+1} |\n')
        local_core_weights = []
        local_loss = []

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        core_weights = []
        avg_test_acc = 0
        for idx in idxs_users:
            print(f'Start for User {idx}:\n')
            print('The user {} model is {}'.format(idx, user_local_model_name[idx]))
            user_local_model[idx].small_model = copy.deepcopy(core_model)

            core_model_dict, _ = user_local_model[idx].update_weights(global_round=epoch)
            local_core_weights.append(copy.deepcopy(core_model_dict))
            test_acc, _ = user_local_model[idx].inference()
            print('|------ User {} Test ACC: {:.2f}%'.format(idx, 100 * test_acc))
            avg_test_acc += test_acc
        avg_test_acc = avg_test_acc / m
        print('After all user local training...')
        print("|---- Avg local test ACC: {:.2f}%".format(100 * avg_test_acc))
        test_acc_list.append(avg_test_acc)

        print('start global avg...')
        global_weights = average_weights(local_core_weights)
        core_model.load_state_dict(global_weights)

    
    print('Communication Round {} finish'.format(args.communication_round))
    print('Start All the local final test...')
    core_model.eval()
    
    test_acc_final = 0
    for idx in range(args.num_users):            
        print(f'| --- Start for User {idx}:\n')

        _, _ = user_local_model[idx].update_weights(global_round=epoch)
        test_acc, _ = user_local_model[idx].inference()
        print('|------ User {} Test ACC: {:.2f}%'.format(idx, 100 * test_acc))
        test_acc_final += test_acc
    avg_test_acc_final = test_acc_final / args.num_users
    print("Final avg test acc: {:.2f}%".format(100 * avg_test_acc_final))



    folder = './save/our_dataset_{}_new'.format(args.dataset)
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_name = os.path.join(folder, 'fed_pretrained_{}_cr_{}_user_{}_frac_{}_iid_{}_fun_{}_ab_{}_results.txt'.format(
        args.pre_trained, args.communication_round, args.num_users, args.frac, args.iid, args.lambda_function, args.ablation))
    with open(file_name, 'w') as f:
        for item in test_acc_list:
            f.write("{}".format(item))
            f.write('\n')

        f.write('avg_test_acc_final: {:.2f}%'.format(100 * avg_test_acc_final))

    
    print(exp_details(args))
    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
    print('during updates, avg_tes_acc_list is {}'.format(test_acc_list))
    print('avg_test_acc_final is {}'.format(avg_test_acc_final))
    












    
   


    