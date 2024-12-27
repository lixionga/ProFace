import sys
sys.path.append('..')
import torch.nn as nn
import torch
import numpy as np

class LatentMapper(nn.Module):
    def __init__(self):
        super().__init__()
        slope = 0.2
        self.model = nn.Sequential(
            # nn.Linear(2560, 2048),
            # nn.LeakyReLU(negative_slope=slope),
            # # nn.BatchNorm1d(n_hid, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.Linear(2048,1024),
            # nn.LeakyReLU(negative_slope=slope),
            # nn.BatchNorm1d(n_hid // 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.Linear(1088, 1024),
            # nn.LeakyReLU(negative_slope=slope),
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=slope),
            nn.Linear(512, 512)
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a= slope)
                nn.init.constant_(m.bias, 0)
        
        
        # self.fc_net = []
        # self.fc_net +=  [nn.Linear(8448,8192)]
        # self.fc_net +=  [nn.LeakyReLU(negative_slope=slope)]
        # num_down_sample = 9
        # for i in range(num_down_sample):
        #     self.fc_net += [nn.Linear(2**(13-i),2**(13-i-1))]
        #     self.fc_net += [nn.LeakyReLU(negative_slope=slope)]
        # self.fc_net +=  [nn.Linear(16,10)]
        # self.fc_net +=  [nn.LeakyReLU(negative_slope=slope)]
        # self.fc_net = nn.Sequential(*self.fc_net)
            
    


    def forward(self, input_tensor):
        return self.model(input_tensor)
    

class LatentMapperPSP(nn.Module):
    def __init__(self):
        super().__init__()
        slope = 0.2
        self.model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.LeakyReLU(negative_slope=slope),
            # nn.BatchNorm1d(n_hid, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Linear(2048,4096),
            nn.LeakyReLU(negative_slope=slope),
            # nn.BatchNorm1d(n_hid // 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(negative_slope=slope),
            nn.Linear(2048, 1024),
            # nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=slope),
            nn.Linear(1024, 512)
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a= slope)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_tensor):
        return self.model(input_tensor)
    
class FusionMapper(nn.Module):
    def __init__(self):
        super().__init__()
        slope = 0.2
        self.model = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=slope),
            nn.Linear(512, 512)
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a= slope)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_tensor):
        return self.model(input_tensor)
    
class FusionMapper10(nn.Module):
    def __init__(self):
        super().__init__()
        slope = 0.2
        self.model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(negative_slope=slope),
            nn.Linear(1024, 2048),            
            nn.LeakyReLU(negative_slope=slope),
            nn.Linear(2048, 1024),            
            nn.LeakyReLU(negative_slope=slope),
            nn.Linear(1024, 512)
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a= slope)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_tensor):
        return self.model(input_tensor)

class FusionMapper19(nn.Module):
    def __init__(self):
        super().__init__()
        slope = 0.2
        self.model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(negative_slope=slope),
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),            
            nn.LeakyReLU(negative_slope=slope),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),            
            nn.LeakyReLU(negative_slope=slope),
            nn.Linear(1024, 512)
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a= slope)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_tensor):
        return self.model(input_tensor)



class InversionMapper(nn.Module):
    def __init__(self):
        super().__init__()
        slope = 0.2
        self.model = nn.Sequential(
            nn.Linear(640, 512),
            nn.LeakyReLU(negative_slope=slope),
            nn.Linear(512, 512)
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a= slope)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_tensor):
        return self.model(input_tensor)


class Passwords_Mapper(nn.Module):
    def __init__(self):
        super().__init__()
        slope = 0.2
        self.model = nn.Sequential(
            nn.Linear(640, 512),
            nn.LeakyReLU(negative_slope=slope),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Linear(512, 512)
            # nn.LeakyReLU(negative_slope=slope),
            # # nn.BatchNorm1d(n_hid, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.Linear(256, 128),
            # nn.LeakyReLU(negative_slope=slope),
            # # nn.BatchNorm1d(n_hid, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.Linear(128, 64)


        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a= slope)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_tensor):
        return self.model(input_tensor)


if __name__ == '__main__':
    # id = np.random.rand(8, 14, 1024)
    # id = torch.from_numpy(id)
    # id = id.to(torch.float32)
    # id = id.to(Global_Config.device)
    # net = LatentMapper()
    # net = net.to(Global_Config.device)
    # with torch.no_grad():
    #     out = net(id)
    #
    # print(f'out type is {type(out)}')
    # print(f'out shape is{out.shape}')
    # # print(f'out  is{a}')

    # input = torch.randint(low=0, high=2, size=(16,1), out=None, requires_grad=False)
    # print(f'{input}')
    # print(f'{torch.sum(input)}')

    input = torch.rand((2,2))
    print(f'{input}')
    print(f'{torch.sum(input)}')

    

