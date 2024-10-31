
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, GAE, GATv2Conv, GraphSAGE, GENConv, GMMConv, \
    GravNetConv, MessagePassing, global_max_pool, global_add_pool, GAT, GINConv, GINEConv, GraphNorm, SAGEConv, RGATConv

from torch_geometric.data import Data


class MetaFluAD(nn.Module):
    def __init__(self, cnn_outdim=256):
        super(MetaFluAD, self).__init__()

        self.SE_CNN = SE_CNN(1, cnn_outdim)
        self.transformer = TransformerModel(cnn_outdim,  # T_input_dim,
                                            512,  # T_hidden_dim,
                                            2,  # T_num_layers,
                                            4,  # T_num_heads,
                                            64  # T_output_dim
                                            )
        self.ResGAT1 = MultiGAT(cnn_outdim, 128 // 2, 128 // 2, 4, 2, concat='True')

        self.regression1 = RegressionModel1(1024, 512, 1)

    def forward(self, data):
        x = data.x

        x = self.SE_CNN(x)

        x_t = self.transformer(x.unsqueeze(1)).squeeze(1)

        data_x = x.squeeze(1)

        x_r = self.ResGAT1(data_x, data.edge_index)

        x = torch.cat((x_t.squeeze(1), x_r), dim=1)

        feature = x
        x, n = create(x, data.edge_index, data.edge_index.shape[1])

        ypre = self.regression1(x)

        return ypre , feature

class RegressionModel0(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RegressionModel0, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc = nn.Linear(input_dim, output_dim, bias = False)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        return x



class MetaFluAD_late_concat(nn.Module): # Late concatenation fusion model
    def __init__(self, metadata_dim,cnn_outdim=256):
        super(MetaFluAD_late_concat, self).__init__()

        self.SE_CNN = SE_CNN(1, cnn_outdim)
        self.transformer = TransformerModel(cnn_outdim,  # T_input_dim,
                                            512,  # T_hidden_dim,
                                            2,  # T_num_layers,
                                            4,  # T_num_heads,
                                            64  # T_output_dim
                                            )
        self.ResGAT1 = MultiGAT(cnn_outdim, 128 // 2, 128 // 2, 4, 2, concat='True')

        self.regression1 = RegressionModel1((512 + metadata_dim)*2, 512 + metadata_dim, 1) # Add dims for metadata

    def forward(self, data):
        x = data.x

        x = self.SE_CNN(x)

        x_t = self.transformer(x.unsqueeze(1)).squeeze(1)

        data_x = x.squeeze(1)

        x_r = self.ResGAT1(data_x, data.edge_index)

        x = torch.cat((x_t.squeeze(1), x_r), dim=1)

        feature = x

        x = torch.cat((x,data.metadata), dim = 1) #concatenate metadata

        x, n = create(x, data.edge_index, data.edge_index.shape[1]) #concatenate feature vector of nodes

        ypre = self.regression1(x)

        return ypre , feature
    
class MetaFluAD_early_attention(nn.Module): # Early attention fusion model
    def __init__(self, metadata_dim, cnn_outdim = 256):
        super(MetaFluAD_early_attention, self).__init__()

        self.SE_CNN = SE_CNN(1,cnn_outdim)
        self.transformer = TransformerModel(cnn_outdim, 512, 2, 4, 64)
        self.GAT = MultiGAT(cnn_outdim, 128 // 2, 128 // 2, 4, 2, concat = 'True')
        self.regression1 = RegressionModel1(512*2,512,1)
        self.regression2 = RegressionModel0(metadata_dim, cnn_outdim)

    def forward(self, data):
        x = data.x

        x = self.SE_CNN(x)

        mask = self.regression2(data.metadata) # create 256 dim vector from metadata

        x = torch.mul(x,mask)

        x_t = self.transformer(x.unsqueeze(1)).squeeze(1)

        data_x = x.squeeze(1)

        x_r = self.GAT(data_x, data.edge_index)

        x = torch.cat((x_t.squeeze(1), x_r), dim = 1)

        feature = x

        x, n = create(x, data.edge_index, data.edge_index.shape[1])

        ypre = self.regression1(x)

        return ypre, feature


class MetaFluAD_metaGAT(nn.Module):
    def __init__(self, metadata_dim, cnn_outdim = 256):
        super(MetaFluAD_metaGAT, self).__init__()
        self.SE_CNN = SE_CNN(1,cnn_outdim)
        self.transformer = TransformerModel(cnn_outdim, 512, 2, 4, 64)
        self.GAT = MultiGAT(cnn_outdim + metadata_dim, 128 // 2, 128 // 2, 4, 2, concat = 'True')
        self.regression1 = RegressionModel1(512*2,512,1)

    def forward(self, data):
        x = data.x

        x = self.SE_CNN(x)

        x_t = self.transformer(x.unsqueeze(1)).squeeze(1)

        # Concatenate metadata with feature vec, before entering GAT
        data_x = torch.cat((x.squeeze(1),data.metadata), dim = 1)

        x_r = self.GAT(data_x, data.edge_index)

        x = torch.cat((x_t.squeeze(1), x_r), dim = 1)

        feature = x

        x, n = create(x, data.edge_index, data.edge_index.shape[1])

        ypre = self.regression1(x)

        return ypre, feature
    

class MetaFluAD_noGAT(nn.Module):
    def __init__(self, cnn_outdim=256):
        super(MetaFluAD_noGAT, self).__init__()

        self.SE_CNN = SE_CNN(1, cnn_outdim)
        self.transformer = TransformerModel(cnn_outdim,  # T_input_dim,
                                            512,  # T_hidden_dim,
                                            2,  # T_num_layers,
                                            4,  # T_num_heads,
                                            64  # T_output_dim
                                            )
        # self.ResGAT1 = MultiGAT(cnn_outdim, 128 // 2, 128 // 2, 4, 2, concat='True')

        self.regression1 = RegressionModel1(512, 256, 1)

    def forward(self, data):
        x = data.x

        x = self.SE_CNN(x)

        x_t = self.transformer(x.unsqueeze(1)).squeeze(1)

        data_x = x.squeeze(1)

        x = x_t.squeeze(1)

        feature = x
        x, n = create(x, data.edge_index, data.edge_index.shape[1])

        ypre = self.regression1(x)

        return ypre , feature
    



class SEBlock1(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x).view(x.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        # print("-----out.shape------")
        # print(out.shape)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = x * out
        return out


class CNN1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN1, self).__init__()
        self.in_channel = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool(out)

        return out


class CNN2(nn.Module):
    def __init__(self, in_channels, out_dim):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=(3, 3), stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        # self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(32)
        #
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(64)
        # self.fc = nn.Linear(32 * 8 * 8, out_dim)
        # self.fc = nn.Linear(32 * 42 * 14, out_dim)
        self.fc = nn.Linear(32 * 11 * 16, out_dim)
        # self.fc = nn.Linear(3456, out_dim)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool(out)


        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.pool(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.pool(out)



        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class SECNN(nn.Module):
    def __init__(self, in_channels, out_dim, reduction_ratio=16):
        super(SECNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.se_block = SEBlock1(32, reduction_ratio)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = self.relu = nn.ReLU(inplace=True)
        self.cnn = CNN2(32, out_dim)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        out = self.se_block(x)
        out = self.cnn(out)
        return out


class SE_CNN(nn.Module):
    def __init__(self, in_channels, out_dim, reduction_ratio=16):
        super(SE_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.se_block1 = SEBlock1(16, reduction_ratio)
        self.se_block2 = SEBlock1(32, reduction_ratio)
        # self.se_block3 = SEBlock1(32, reduction_ratio)

        self.relu = self.relu = nn.ReLU(inplace=True)
        self.cnn1 = CNN1(16, 16)
        self.cnn2 = CNN1(16, 32)
        self.cnn3 = CNN1(32, 32)
        self.cnn4 = CNN1(32, 32)
        self.bn1 = nn.BatchNorm2d(16)
        # self.fc = nn.Linear(32 * 11 * 16, out_dim)
        self.fc = nn.Linear(32 * 22 * 8, out_dim)

    def forward(self, x):

        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)

        out = self.se_block1(x)
        out = self.cnn1(out)
        out = self.cnn2(out)

        out = self.se_block2(out)
        out = self.cnn3(out)
        out = self.cnn4(out)
        # print("+++++++++++")
        # print(out.shape)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = F.relu(out)
        return out


class GCN(torch.nn.Module):
    def __init__(self, in_channel, hidden_channels, out_channel):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channel, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channel)
        self.fc = nn.Linear(in_channel, out_channel)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, edge_index):
        res = self.fc(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.relu(x+res)

class RESGATv2(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers, num_heads, dropout=0.3):
        super(RESGATv2, self).__init__()
        self.dropout = dropout

        self.gat_layers = nn.ModuleList()

        self.gat_layers.append(GATv2Conv(in_features, hidden_features, heads=num_heads[0]))

        for l in range(1, num_layers - 1):
            self.gat_layers.append(GATv2Conv(hidden_features * num_heads[l - 1], hidden_features, heads=num_heads[l]))

        self.gat_layers.append(GATv2Conv(hidden_features * num_heads[-2], out_features, heads=num_heads[-1]))

        self.residual_layer = nn.Linear(in_features, out_features * num_heads[-1])

    def forward(self, x, edge_index):
        # x = data.x
        x_res = self.residual_layer(x)
        for layer in self.gat_layers[:-1]:
            x = layer(x, edge_index).flatten(1)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gat_layers[-1](x, edge_index).flatten(1)
        x = F.elu(x)

        x = F.relu(x + x_res)
        return x


class MultiGAT(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, num_layers, concat='True'):
        super(MultiGAT, self).__init__()
        self.num_layers = num_layers

        # Define the input layer.
        self.conv1 = GATConv(in_channels,
                             hidden_channels,
                             concat=concat,
                             heads=num_heads,
                             dropout=0.2,
                             bias=True)

        # Define the output layer.
        self.convN = GATConv(
            hidden_channels * num_heads,
            out_channels,
            concat=concat,
            dropout=0.2,
            heads=num_heads)

        self.fc = nn.Linear(in_channels, num_heads * out_channels)

    def forward(self, x, edge_index):
        res = self.fc(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.convN(x, edge_index)
        x = F.relu(x + res)
        return x


class RegressionModel1(nn.Module):
    def __init__(self, input_dim, reg_hidden_dim, output_dim):
        super(RegressionModel1, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = reg_hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, reg_hidden_dim, bias=True)
        # self.fc2 = nn.Linear(reg_hidden_dim, reg_hidden_dim, bias=False)
        # self.fc3 = nn.Linear(reg_hidden_dim, reg_hidden_dim, bias=True)
        self.fc4 = nn.Linear(reg_hidden_dim, output_dim)

        self.dropout1 = nn.Dropout(p=0.5)


    def forward(self, x):
        # x = F.relu(self.fc1(x))
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout1(x)

        x = self.fc4(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 num_layers, num_heads, output_dim):
        super().__init__()
        #
        #
        """
        编码器   nn.TransformerEncoder
        encoder_layer：用于构造编码器层的类，默认为 nn.TransformerEncoderLayer。
        num_layers：编码器层的数量。默认值为 6。
        norm：归一化模块的类，用于在每个编码器层之间进行归一化，默认为 nn.LayerNorm。
        batch_first：输入张量是否以 batch 维度为第一维。默认为 False。
        dropout：每个编码器层输出之前的 dropout 概率。默认值为 0.1

        编码器层  nn.TransformerEncoderLayer
        d_model：输入特征的维度和输出特征的维度。默认值为 512。
        nhead：多头注意力的头数。默认值为 8。
        dim_feedforward：前馈神经网络的隐藏层大小。默认值为 2048。
        dropout：每个子层输出之前的 dropout 概率。默认值为 0.1。
        activation：前馈神经网络中使用的激活函数类型。默认值为 'relu'。
        normalize_before：是否在每个子层之前进行层归一化。默认值为 False。
        """
        self.transformer_encoder_layer = \
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads,
                                       dim_feedforward=hidden_dim,
                                       dropout=0.2, activation='relu',
                                       batch_first=True,
                                       # normalize_before=True
                                       )
        self.transformer = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers,
            # dropout=0.2
        )
        self.fc1 = nn.Linear(input_dim, num_heads * output_dim)
        # self.dropout1 = nn.Dropout(p=0.3)
        # self.norm = nn.LayerNorm(output_dim)

    def Resforward(self, x):
        se = x
        x = self.transformer(x)
        x = F.relu(x + se)
        x = self.transformer(x)
        # x = x.squeeze(1)  # 去除序列长度为1的维度
        # x = self.fc(x + se)  # 将Transformer的输出转换为256维向量
        x = self.dropout1(x)
        x = self.fc(x)
        # x = F.relu(x+se)
        # x = self.fc(x)
        return x

    def forward(self, x):
        input = self.fc1(x)
        # x: (batch_size, sequence_length, input_size)
        x = x.transpose(0, 1)  # 将序列长度放到第一维，变成 (sequence_length, batch_size, input_size)
        x = self.transformer(x)  # Transformer 编码器

        x = x.transpose(0, 1)  # 将序列长度放回到第二维，变成 (batch_size, sequence_length, input_size)
        x = F.relu(x + input)
        # x = x + input
        # x = self.norm(x)
        # x = self.fc(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        se = self.squeeze(x).view(batch_size, channels)
        se = self.excitation(se).view(batch_size, channels, 1, 1)
        return x * se


def create(x, edge_index, num_edge):
    features = []
    for i in range(num_edge):
        # print(edge_index)
        m = edge_index[0][i]
        n = edge_index[1][i]
        # print(m, n)
        # print(x[m], x[n])
        feature = torch.flatten(torch.cat((x[m], x[n])))
        # feature = torch.flatten((x[m] + x[n]))
        # print(feature)
        features.append(feature)
        # if i == 0:
        #     a = feature
        #     continue
        # a = torch.cat(a, feature.unsqueeze(0), dim=0)

    return torch.stack(features), features