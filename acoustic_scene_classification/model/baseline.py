import torch
import torch.nn.functional as F
from torch import nn

class resnet_layer(nn.Module) :
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, learn_bn=True, use_relu = True) :
        super(resnet_layer, self).__init__()
        self.use_relu = use_relu
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        torch.nn.init.kaiming_normal_(self.conv.weight)
        self.bn = nn.BatchNorm2d(in_channels, affine=learn_bn)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x) :
        x = self.bn(x)
        if self.use_relu :
            x = self.relu(x)
        x = self.conv(x)
        return x

class resnet_module(nn.Module) :
    def __init__(self, num_filters_in, num_filters, stack, res_block) :
        super(resnet_module, self).__init__()
        self.stack = stack
        self.res_block = res_block

        strides = (1, 1)
        if stack > 0 and res_block == 0:
            strides = (1, 2)
        self.conv1_path1 = resnet_layer(num_filters_in, num_filters, (3, 3), strides, (1, 1), learn_bn=False, use_relu=True)
        self.conv2_path1 = resnet_layer(num_filters, num_filters, (3, 3), (1,1), (1, 1), learn_bn=False, use_relu=True)
        self.conv1_path2 = resnet_layer(num_filters_in, num_filters, (3, 3), strides, (1, 1), learn_bn=False, use_relu=True)
        self.conv2_path2 = resnet_layer(num_filters, num_filters, (3, 3), (1,1), (1, 1), learn_bn=False, use_relu=True)

        if stack > 0 and res_block == 0:
            self.pool_path1 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
            self.pool_path2 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))


    def forward(self, res_path1, res_path2) :
        conv_path1 = self.conv1_path1(res_path1)
        conv_path1 = self.conv2_path1(conv_path1)
        conv_path2 = self.conv1_path2(res_path2)
        conv_path2 = self.conv2_path2(conv_path2)
        if self.stack > 0 and self.res_block == 0:
            res_path1 = self.pool_path1(res_path1)
            res_path2 = self.pool_path2(res_path2)

            pad_path1 = torch.zeros(res_path1.size()).cuda()
            pad_path2 = torch.zeros(res_path2.size()).cuda()
            res_path1 = torch.cat((res_path1, pad_path1), dim=1)
            res_path2 = torch.cat((res_path2, pad_path2), dim=1)
        res_path1 = conv_path1 + res_path1
        res_path2 = conv_path2 + res_path2

        return res_path1, res_path2


class model_resnet(nn.Module) :
    def __init__(self, num_class, num_filters=24, num_stacks=4, output_num_filter_factor=1, stacking_frame=None, domain_aux=False) :
        super(model_resnet, self).__init__()
        self.num_stacks = num_stacks
        self.output_num_filter_factor = output_num_filter_factor
        self.stacking_frame = stacking_frame
        self.domain_aux = domain_aux
        self.num_res_blocks = 2

        if self.stacking_frame :
            self.sf_conv = nn.Conv2d(3, 3, kernel_size=(1,1), stride=(1,stacking_frame), padding=(0,0))
        self.resnet_path1 = resnet_layer(3, num_filters, (3,3), (1,2), (1,1),learn_bn=True, use_relu=False)
        self.resnet_path2 = resnet_layer(3, num_filters, (3,3), (1,2), (1,1),learn_bn=True, use_relu=False)

        self.resnet_module_list = nn.ModuleList()
        for stack in range(self.num_stacks) :
            for res_block in range(self.num_res_blocks) :
                if stack > 0 and res_block == 0 :
                    num_filters_in = int(num_filters / 2)
                else :
                    num_filters_in = num_filters

                self.resnet_module_list.append(resnet_module(num_filters_in, num_filters, stack, res_block))
            num_filters *= 2
        num_filters = int(num_filters / 2)
        self.resnet_out1 = resnet_layer(num_filters, num_filters*output_num_filter_factor, (1,1), (1,1), (0,0), learn_bn=True, use_relu=True)
        self.resnet_out2 = resnet_layer(num_filters, num_class, (1,1), (1,1), (0,0), learn_bn=True, use_relu=False)
        self.bn_out = nn.BatchNorm2d(num_class, affine=True)

    def forward(self, x) :
        if self.stacking_frame :
            x = self.sf_conv(x)
        Split1 = x[:,:,:64,:]
        Split2 = x[:,:,64:,:]

        ResidualPath1 = self.resnet_path1(Split1)
        ResidualPath2 = self.resnet_path2(Split2)
        for i in range(len(self.resnet_module_list)) :
            ResidualPath1, ResidualPath2 = self.resnet_module_list[i](ResidualPath1, ResidualPath2)
        ResidualPath = torch.cat((ResidualPath1, ResidualPath2), dim=2)

        OutputPath = self.resnet_out1(ResidualPath)
        OutputPath = self.resnet_out2(OutputPath)
        OutputPath = self.bn_out(OutputPath)
        OutputPath = F.adaptive_avg_pool2d(OutputPath, (1,1))
        OutputPath = F.softmax(OutputPath, dim=1).squeeze()

        return OutputPath