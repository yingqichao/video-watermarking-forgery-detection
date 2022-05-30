import torchvision


class DefaultConfig(object):
    cuda_index = "1"
    attack_target = 'untargeted'
    attack_method = 'IGSM'
    model_name = 'resnet50'
    my_model = torchvision.models.resnet50(pretrained=True)
    if attack_method == 'FGSM':
        myepsilon = 3 / 255
        steps = 1
    elif attack_method == 'IGSM':
        if attack_target == 'untargeted':
            myepsilon = 3 / 255
            steps = 10
        elif attack_target == 'targeted':
            myepsilon = 7 / 255
            steps = 10
    elif attack_method == 'CW':
        steps = 50
        binary_search_steps = 9
        stepsize = 0.01
        abort = True
        initial_const = 1e-3
    quality_list = [70]
    compress_needlist = 1
    if compress_needlist == 1:
        dist_para = 0.2
        max_iters = 10
        learning_rate = 0.1
        miu = 0.9
        eta_list = [
            0.2
        ]

    def show(self):
        print('cuda_index:', self.cuda_index)
        print('model:', self.model_name)
        if self.compress_needlist == 1 or self.compress_needlist == 0:
            print('dist_para:', self.dist_para)
            print('max_iters:',self.max_iters)
            print('learning_rate:',self.learning_rate)
            print('eta_list:', self.eta_list)
        print('attack_target:', self.attack_target)
        print('attack_method:', self.attack_method)
        if self.attack_method == 'FGSM' or self.attack_method == 'IGSM':
            print('myepsilon:', self.myepsilon * 255)
            print('steps:', self.steps)
        elif self.attack_method == 'CW':
            print('steps:',self.steps)
            print('binary_search_steps:',self.binary_search_steps)
            print('stepsize:',self.stepsize)
            print('abort:',self.abort)
            print('initial_const:',self.initial_const)
        print('quality_list:', self.quality_list)
        
