
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
        Encoder: nn.TransformerEncoder
        encoder_layer: The class used to construct the encoder layer, defaulting to nn.TransformerEncoderLayer.
        num_layers: The number of encoder layers. Default is 6.
        norm: The normalization module class, used to normalize between each encoder layer, defaulting to nn.LayerNorm.
        batch_first: Whether the input tensor has the batch dimension as the first dimension. Default is False.
        dropout: The dropout probability before each encoder layer’s output. Default is 0.1.

        Encoder Layer: nn.TransformerEncoderLayer
        d_model: The dimension of input features and output features. Default is 512.
        nhead: The number of heads in the multi-head attention mechanism. Default is 8.
        dim_feedforward: The size of the hidden layer in the feedforward neural network. Default is 2048.
        dropout: The dropout probability before each sub-layer’s output. Default is 0.1.
        activation: The type of activation function used in the feedforward neural network. Default is 'relu'.
        normalize_before: Whether to apply layer normalization before each sub-layer. Default is False.
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
        # x = x.squeeze(1)  # Remove dimensions with a sequence length of 1
        # x = self.fc(x + se)  # Convert Transformer output to a 256-dimensional vector
        x = self.dropout1(x)
        x = self.fc(x)
        # x = F.relu(x+se)
        # x = self.fc(x)
        return x

    def forward(self, x):
        input = self.fc1(x)
        # x: (batch_size, sequence_length, input_size)
        x = x.transpose(0, 1)  # Move sequence length to the first dimension, becomes (sequence_length, batch_size, input_size)
        x = self.transformer(x)  # Transformer encoder

        x = x.transpose(0, 1)  # Move sequence length back to the second dimension, becomes (batch_size, sequence_length, input_size)
        x = F.relu(x + input)
        # x = x + input
        # x = self.norm(x)
        # x = self.fc(x)
        return x


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


class Smart_tiny(nn.Module):
    """
    Tiny version of the MetaFluAD model, the transformer and GAT are removed, and the output
    dimension of the CNN is much smaller.
    """
    def __init__(self, cnn_outdim=32):
        super(Smart_tiny, self).__init__()

        self.SE_CNN = CNN_tiny(1, cnn_outdim)
        self.regression1 = RegressionModel1(cnn_outdim, cnn_outdim//2, 1)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, data):
        x = data.x
        x = self.SE_CNN(x, data.edge_index, data.edge_index.shape[1])
        x = F.leaky_relu(x, negative_slope=0.1)
        feature = x
        x = self.dropout1(x)
        ypre =self.regression1(x)
        return ypre , feature
    

class CNN_tiny(nn.Module):
    def __init__(self, in_channels, out_dim, reduction_ratio=16):
        super(CNN_tiny, self).__init__()
        self.se_block2 = SEBlock1(4, reduction_ratio)
        self.se_block3 = SEBlock1(8, reduction_ratio)

        self.conv1 = CNN_custom(in_channels,4,3,1,2,1,2)
        self.conv1b = CNN_custom(in_channels,4,3,1,2,1,2)
        self.conv2 = CNN_custom(4,4,3,1,2,1,2)
        self.conv2b = CNN_custom(4,4,3,1,2,1,2)

        self.conv3 = CNN_custom(4,4,3,1,2,1,2)
        self.conv4 = CNN_custom(4,8,3,1,2,1,2)
        self.conv5 = CNN_custom(8,8,3,1,2,1,2)
        self.conv6 = CNN_custom(8,8,3,1,3,1,3)

        self.fc = nn.Linear(64, out_dim)

    def forward(self, x, e, s):
        x = create_cnnstack2(x, e, s)

        x1 = self.conv1(x[:,0].unsqueeze(1))
        x2 = self.conv1b(x[:,1].unsqueeze(1))
        x1 = self.conv2(x1)
        x2 = self.conv2b(x2)
        out = x1 - x2

        out = self.conv3(out)
        out = self.se_block2(out)
        out = self.conv4(out)

        out = self.conv5(out)
        out = self.se_block3(out)
        out = self.conv6(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
    

class CNN_custom(nn.Module):
    def __init__(self, in_channels, out_channels, k_conv, pad_conv, k_pool, pad_pool, str_pool):
        super(CNN_custom, self).__init__()
        self.in_channel = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=k_conv, stride=1, padding=pad_conv)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.pool = nn.MaxPool2d(kernel_size=k_pool, stride=str_pool, padding=pad_pool)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool(out)
        return out
    
def create_cnnstack2(x, edge_index, num_edge):
    features = []
    for i in range(num_edge):
        m = edge_index[0][i]
        n = edge_index[1][i]
        feature = torch.cat((x[m] ,x[n]))
        features.append(feature)

    return torch.stack(features)