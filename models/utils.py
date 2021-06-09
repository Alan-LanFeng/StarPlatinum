import torch
import torch.nn as nn
from torch.autograd import Variable
from copy import deepcopy
import math


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed

    def forward(self, src, tgt, src_mask, tgt_mask, query_pos=None):
        """
        Take in and process masked src and target sequences.
        """
        output = self.encode(src, src_mask)
        return self.decode(output, src_mask, tgt, tgt_mask, query_pos)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask, query_pos=None):
        return self.decoder(tgt, memory, src_mask, tgt_mask, query_pos)


class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers
    """

    def __init__(self, layer, n):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, x_mask):
        """
        Pass the input (and mask) through each layer in turn.
        """
        for layer in self.layers:
            x = layer(x, x_mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward (defined below)
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        Follow Figure 1 (left) for connections.
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.
    """

    def __init__(self, layer, n, return_intermediate=False):
        super(Decoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = nn.LayerNorm(layer.size)
        self.return_intermediate = return_intermediate

    def forward(self, x, memory, src_mask, tgt_mask, query_pos=None):

        intermediate = []

        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask, query_pos)

            if self.return_intermediate:
                intermediate.append(self.norm(x))

        if self.norm is not None:
            x = self.norm(x)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(x)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return x


class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward (defined below)
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    # TODO How to fusion the feature
    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, x, memory, src_mask, tgt_mask, query_pos=None):
        """
        Follow Figure 1 (right) for connections.
        """
        m = memory
        q = k = self.with_pos_embed(x, query_pos)
        x = self.sublayer[0](x, lambda x: self.self_attn(q, k, x, tgt_mask))
        x = self.with_pos_embed(x, query_pos)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """
        Take in model size and number of heads.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        #  We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model, bias=True), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Implements Figure 2
        """
        if len(query.shape) > 3:
            batch_dim = len(query.shape) - 2
            batch = query.shape[:batch_dim]
            mask_dim = batch_dim
        else:
            batch = (query.shape[0],)
            mask_dim = 1
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(dim=mask_dim)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(*batch, -1, self.h, self.d_k).transpose(-3, -2) for l, x in
                             zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(-3, -2).contiguous().view(*batch, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PointerwiseFeedforward(nn.Module):
    """
    Implements FFN equation.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PointerwiseFeedforward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias=True)
        self.w_2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.relu(self.w_1(x))))


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        """
        return x + self.dropout(sublayer(self.norm(x)))


def clones(module, n):
    """
    Produce N identical layers.
    """
    assert isinstance(module, nn.Module)
    return nn.ModuleList([deepcopy(module) for _ in range(n)])


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'
    """
    d_k = query.size(-1)

    # Q,K,V: [bs,h,num,dim]
    # scores: [bs,h,num1,num2]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # mask: [bs,1,1,num2] => dimension expansion

    if mask is not None:
        scores = scores.masked_fill_(mask == 0, value=-1e9)
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class LinearEmbedding(nn.Module):
    def __init__(self, inp_size, d_model):
        super(LinearEmbedding, self).__init__()
        # lut => lookup table
        self.lut = nn.Linear(inp_size, d_model, bias=True)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Implement the PE function.
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        x = x + Variable(self.pe[:x.shape[-2]], requires_grad=False)
        return self.dropout(x)

class TypeEmb(nn.Module):
    """
    Implement the PE function.
    """

    def __init__(self,d_type,d_model):
        super(TypeEmb, self).__init__()
        self.lut = nn.Linear(d_type,d_model,bias=False)
    def add_type(self,type):
        self.type_emb = self.lut(type)
    def forward(self, x):
        x = x + self.type_emb
        return x


class PosYawEmbed(nn.Module):
    """
    Implement the PE function.
    """
    def __init__(self,d_model):
        super(PosYawEmbed, self).__init__()
        # self.dropout = nn.Dropout(p=dropout)
        #
        self.pos_emb = nn.Linear(2, d_model, bias=True)
        self.yaw_emb = nn.Linear(2, d_model, bias=True)
    def add_py(self,pos,yaw):
        self.pos = pos
        self.yaw = yaw
    def forward(self, x):
        pos = self.pos_emb(self.pos)
        yaw = self.yaw_emb(self.yaw)
        x = x + pos + yaw
        return x




class FeedForwardWarrper(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super(FeedForwardWarrper, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear_proj = nn.Linear(self.input_size, self.output_size)
        self.FFN = PointerwiseFeedforward(self.output_size, self.hidden_size, dropout)
        self.residual_block = SublayerConnection(output_size, dropout)
        self.in_norm = nn.LayerNorm(output_size)
        self.out_norm = nn.LayerNorm(output_size)

    def forward(self, x):
        proj_feature = self.in_norm(self.linear_proj(x))
        output = self.residual_block(proj_feature, self.FFN)
        return self.out_norm(output)


class FeedForwardWarrperAdd(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super(FeedForwardWarrperAdd, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.FFN = PointerwiseFeedforward(self.output_size, self.hidden_size, dropout)
        self.residual_block = SublayerConnection(output_size, dropout)
        self.out_norm = nn.LayerNorm(output_size)

    def forward(self, x):
        output = self.residual_block(x, self.FFN)
        return self.out_norm(output)


class RegressionGenerator(nn.Module):

    def __init__(self, input_size, hidden_size, output_size=2, dropout=0.1):
        super(RegressionGenerator, self).__init__()
        self.reg_mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size, bias=True),
            nn.Linear(hidden_size, output_size, bias=True))

    def forward(self, x):
        pred = self.reg_mlp(x)
        pred = pred.cumsum(dim=-2)
        return pred


class ModalityGenerator(nn.Module):
    def __init__(self, input_size, modality_num=6):
        super(ModalityGenerator, self).__init__()
        self.modality_num = modality_num
        self.input_size = input_size
        self.expand_proj = nn.Linear(input_size, modality_num * input_size)
        self.linears = clones(nn.Linear(input_size, input_size, bias=True), modality_num)

    def forward(self, x):
        '''
            x:[batchsize, number agents, future_num_frame, dim]
            out:[batchsize, number agents, modality_num, future_num_frame, dim]
        '''
        single_modality_shape = x.shape
        multi_modality = self.expand_proj(x)
        multi_modality = multi_modality.view(*single_modality_shape, self.modality_num)
        multi_modality = multi_modality.permute(4, 0, 1, 2,
                                                3)  # [modality_num, batchsize, number agents, future_num_frame, dim]
        multi_modality = [self.linears[i](multi_modality[i]) for i in range(self.modality_num)]
        multi_modality = torch.stack(multi_modality,
                                     dim=-3)  # [batchsize, number agents, modality_num, future_num_frame, dim]
        return multi_modality


class GeneratorWithProbilities(nn.Module):
    def __init__(self, d_model, out_size, dropout, reg_h_dim=128, dis_h_dim=128, cls_h_dim=128):
        super(GeneratorWithProbilities, self).__init__()
        self.reg_mlp = nn.Sequential(
            nn.Linear(d_model, reg_h_dim * 2, bias=True),
            nn.LayerNorm(reg_h_dim * 2),
            nn.ReLU(),
            nn.Linear(reg_h_dim * 2, reg_h_dim, bias=True),
            nn.Linear(reg_h_dim, out_size, bias=True))
        self.dis_emb = nn.Linear(2, dis_h_dim, bias=True)
        self.cls_FFN = PointerwiseFeedforward(d_model, 2 * d_model, dropout=dropout)
        # self.sublayer = nn.Sequential(
        #                 nn.Linear(d_model, cls_h_dim*2, bias=True),
        #                 nn.LayerNorm(cls_h_dim*2),
        #                 nn.ReLU(),
        #                 nn.Linear(cls_h_dim*2, cls_h_dim),)
        # self.classification_layer = nn.Sequential(
        #     nn.Linear(d_model, cls_h_dim, bias=True),
        #     nn.LayerNorm(cls_h_dim),
        #     nn.ReLU(),
        #     nn.Linear(cls_h_dim, 1, bias=True),
        #     nn.Sigmoid())
        self.cls_mlp = nn.Sequential(
            nn.Linear(d_model, reg_h_dim*2, bias=True),
            nn.LayerNorm(reg_h_dim*2),
            nn.ReLU(),
            nn.Linear(reg_h_dim*2, reg_h_dim, bias=True),
            nn.Linear(reg_h_dim, 1, bias=True),
            nn.Sigmoid())
        # self.cls_emb_layer = SublayerConnection(dis_h_dim+d_model, dropout)
        # self.cls_opt = nn.Sigmoid()
        # self.cls_opt = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        pred = self.reg_mlp(x)
        pred = pred.view(*pred.shape[0:3], -1, 2)  # .cumsum(dim=-2)
        # return pred
        # cls_h = self.cls_FFN(x)
        # cls_h = self.classification_layer(cls_h).squeeze(dim=-1)
        # conf = self.cls_opt(cls_h)
        conf = self.cls_mlp(x).squeeze(dim=-1)
        return pred, conf


# for 626
class GeneratorWithParallelHeads626(nn.Module):
    def __init__(self, d_model, out_size, dropout, reg_h_dim=128, dis_h_dim = 128, cls_h_dim = 128):
        super(GeneratorWithParallelHeads626, self).__init__()
        self.reg_mlp = nn.Sequential(
            nn.Linear(d_model, reg_h_dim*2, bias=True),
            nn.LayerNorm(reg_h_dim*2),
            nn.ReLU(),
            nn.Linear(reg_h_dim*2, reg_h_dim, bias=True),
            nn.Linear(reg_h_dim, out_size, bias=True))
        self.dis_emb = nn.Linear(2, dis_h_dim, bias=True)
        self.cls_FFN = PointerwiseFeedforward(d_model, 2*d_model, dropout=dropout)
        # self.sublayer = nn.Sequential(
        #                 nn.Linear(d_model, cls_h_dim*2, bias=True),
        #                 nn.LayerNorm(cls_h_dim*2),
        #                 nn.ReLU(),
        #                 nn.Linear(cls_h_dim*2, cls_h_dim),)
        self.classification_layer = nn.Sequential(
                                        nn.Linear(d_model, cls_h_dim),
                                        nn.Linear(cls_h_dim, 1, bias=True))
        # self.cls_emb_layer = SublayerConnection(dis_h_dim+d_model, dropout)
        # self.cls_opt = nn.Sigmoid()
        self.cls_opt = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        pred = self.reg_mlp(x)
        pred = pred.view(*pred.shape[0:3],-1,2) #.cumsum(dim=-2)
        # return pred
        cls_h = self.cls_FFN(x)
        cls_h = self.classification_layer(cls_h).squeeze(dim=-1)
        conf = self.cls_opt(cls_h)
        # conf = cls_h
        return pred, conf

class deeper626(nn.Module):
    def __init__(self, d_model, out_size, dropout, reg_h_dim=128, dis_h_dim = 128, cls_h_dim = 128):
        super(deeper626, self).__init__()
        self.reg_mlp = nn.Sequential(
            nn.Linear(d_model, reg_h_dim*2, bias=True),
            nn.LayerNorm(reg_h_dim*2),
            nn.ReLU(),
            nn.Linear(reg_h_dim*2, reg_h_dim, bias=True),
            nn.LayerNorm(reg_h_dim),
            nn.ReLU(),
            nn.Linear(reg_h_dim, reg_h_dim * 2, bias=True),
            nn.LayerNorm(reg_h_dim * 2),
            nn.ReLU(),
            nn.Linear(reg_h_dim * 2, reg_h_dim, bias=True),
            nn.LayerNorm(reg_h_dim),
            nn.ReLU(),
            nn.Linear(reg_h_dim, out_size, bias=True))


        self.cls_FFN = nn.Sequential(
            nn.Linear(d_model, reg_h_dim*2, bias=True),
            nn.LayerNorm(reg_h_dim*2),
            nn.ReLU(),
            nn.Linear(reg_h_dim*2, reg_h_dim, bias=True),
            nn.LayerNorm(reg_h_dim),
            nn.ReLU(),
            nn.Linear(reg_h_dim, reg_h_dim * 2, bias=True),
            nn.LayerNorm(reg_h_dim * 2),
            nn.ReLU(),
            nn.Linear(reg_h_dim * 2, reg_h_dim, bias=True),
            nn.LayerNorm(reg_h_dim),
            nn.ReLU())

        self.classification_layer = nn.Sequential(
                                        nn.Linear(reg_h_dim, cls_h_dim),
                                        nn.Linear(cls_h_dim, 1, bias=True))
        self.cls_opt = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        pred = self.reg_mlp(x)
        pred = pred.view(*pred.shape[0:3],-1,2) #.cumsum(dim=-2)
        # return pred
        cls_h = self.cls_FFN(x)
        cls_h = self.classification_layer(cls_h).squeeze(dim=-1)
        conf = self.cls_opt(cls_h)
        # conf = cls_h
        return pred, conf


class GeneratorWithInteraction(nn.Module):
    def __init__(self, d_model, out_size, dropout, reg_h_dim=128, dis_h_dim=128, cls_h_dim=128):
        super(GeneratorWithInteraction, self).__init__()
        self.reg_cls = nn.Sequential(
            nn.Linear(d_model * 2, reg_h_dim * 2, bias=True),
            nn.LayerNorm(reg_h_dim * 2),
            nn.ReLU(),
        )
        self.reg_mlp1 = nn.Sequential(
            nn.Linear(d_model, reg_h_dim * 2, bias=True),
            nn.LayerNorm(reg_h_dim * 2),
            nn.ReLU(),
            nn.Linear(reg_h_dim * 2, reg_h_dim, bias=True),
            nn.Linear(reg_h_dim, out_size, bias=True))
        self.reg_mlp2= nn.Sequential(
            nn.Linear(d_model, reg_h_dim * 2, bias=True),
            nn.LayerNorm(reg_h_dim * 2),
            nn.ReLU(),
            nn.Linear(reg_h_dim * 2, reg_h_dim, bias=True),
            nn.Linear(reg_h_dim, out_size, bias=True))
        self.dis_emb = nn.Linear(2, dis_h_dim, bias=True)
        self.cls_FFN = PointerwiseFeedforward(d_model, 2 * d_model, dropout=dropout)
        # self.sublayer = nn.Sequential(
        #                 nn.Linear(d_model, cls_h_dim*2, bias=True),
        #                 nn.LayerNorm(cls_h_dim*2),
        #                 nn.ReLU(),
        #                 nn.Linear(cls_h_dim*2, cls_h_dim),)
        self.classification_layer = nn.Sequential(
            nn.Linear(d_model, cls_h_dim),
            nn.Linear(cls_h_dim, 1, bias=True))
        # self.cls_emb_layer = SublayerConnection(dis_h_dim+d_model, dropout)
        # self.cls_opt = nn.Sigmoid()
        self.cls_opt = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        pred1 = self.reg_mlp1(x[:, 0, :, :]).unsqueeze(1)
        pred2 = self.reg_mlp2(x[:, 1, :, :]).unsqueeze(1)
        pred = torch.cat([pred1, pred2], 1)
        pred = pred.view(*pred.shape[0:3], -1, 2)  # .cumsum(dim=-2)
        # return pred
        x = torch.transpose(x, 1, 2)
        x = x.reshape(*x.shape[:2], -1)
        x = self.reg_cls(x)
        cls_h = self.cls_FFN(x)
        cls_h = self.classification_layer(cls_h).squeeze(dim=-1)
        conf = self.cls_opt(cls_h)
        # conf = cls_h
        return pred, conf

class GeneratorWithInteraction1(nn.Module):
    def __init__(self, d_model, out_size, dropout, reg_h_dim=128, dis_h_dim=128, cls_h_dim=128):
        super(GeneratorWithInteraction1, self).__init__()
        self.reg_cls = nn.Sequential(
            nn.Linear(d_model * 2, reg_h_dim * 2, bias=True),
            nn.LayerNorm(reg_h_dim * 2),
            nn.ReLU(),
        )
        self.reg_mlp1 = nn.Sequential(
            nn.Linear(d_model, reg_h_dim * 2, bias=True),
            nn.LayerNorm(reg_h_dim * 2),
            nn.ReLU(),
            nn.Linear(reg_h_dim * 2, reg_h_dim, bias=True),
            nn.Linear(reg_h_dim, out_size, bias=True))
        self.reg_mlp2 = nn.Sequential(
            nn.Linear(d_model, reg_h_dim * 2, bias=True),
            nn.LayerNorm(reg_h_dim * 2),
            nn.ReLU(),
            nn.Linear(reg_h_dim * 2, reg_h_dim, bias=True),
            nn.Linear(reg_h_dim, out_size, bias=True))

        self.cls1 = nn.Sequential(
            PointerwiseFeedforward(d_model, 2 * d_model, dropout=dropout),
            nn.Linear(d_model, cls_h_dim),
            nn.Linear(cls_h_dim, 1, bias=True),
            nn.LogSoftmax(dim=-1))

        self.cls2 = nn.Sequential(
            PointerwiseFeedforward(d_model, 2 * d_model, dropout=dropout),
            nn.Linear(d_model, cls_h_dim),
            nn.Linear(cls_h_dim, 1, bias=True),
            nn.LogSoftmax(dim=-1))

    def forward(self, x):
        pred1 = self.reg_mlp1(x[:,0,:,:]).unsqueeze(1)
        pred2 = self.reg_mlp2(x[:,1,:,:]).unsqueeze(1)
        conf1 = self.cls1(x[:, 0, :, :]).unsqueeze(1)
        conf2 = self.cls2(x[:, 1, :, :]).unsqueeze(1)
        pred = torch.cat([pred1,pred2],1)
        conf = torch.cat([conf1, conf2], 1)
        return pred, conf


class GeneratorWithParallelHeads(nn.Module):
    def __init__(self, d_model, out_size, dropout, reg_h_dim=128, region_proposal_num=6):
        super(GeneratorWithParallelHeads, self).__init__()
        self.reg_mlp = nn.Sequential(
            nn.Linear(d_model, reg_h_dim * 2, bias=True),
            nn.ReLU(),
            nn.Linear(reg_h_dim * 2, reg_h_dim, bias=True),
            nn.ReLU(),
            nn.Linear(reg_h_dim, out_size, bias=True))
        # self.dis_emb = nn.Linear(2, dis_h_dim, bias=True)
        self.cls_FFN = PointerwiseFeedforward(d_model, 2 * d_model, dropout=dropout)
        self.classification_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=True),
            nn.Linear(d_model // 2, 1, bias=True))
        self.cls_opt = nn.Softmax(dim=-1)

    def forward(self, x):
        pred = self.reg_mlp(x)
        pred = pred.view(*pred.shape[:-1], -1, 2).cumsum(dim=-2)
        # endpoint = pred[...,-1,:].squeeze(dim=-2).detach()
        # x = torch.cat((x, endpoint), dim=-1)
        cls_h = self.cls_FFN(x)
        cls_h = self.classification_layer(cls_h).squeeze(dim=-1)
        conf = self.cls_opt(cls_h)
        return pred, conf


class ClassificationHead(nn.Module):
    def __init__(self, d_model):
        super(ClassificationHead, self).__init__()
        self.cls_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2, bias=True),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model, bias=True),
            nn.ReLU(),
            nn.Linear(d_model, 1, bias=True))

    def forward(self, x):
        cls_out = self.cls_mlp(x).squeeze(dim=-1)
        return cls_out



class MLP(nn.Module):
    def __init__(self, in_channels, hidden_unit, verbose=False):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class LaneNet(nn.Module):
    def __init__(self, in_channels, hidden_unit, num_subgraph_layers):
        super(LaneNet, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(f'lmlp_{i}', MLP(in_channels, hidden_unit))
            in_channels = hidden_unit * 2

    def forward(self, lane):
        x = lane
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                # x [bs,max_lane_num,9,dim]
                x = layer(x)
                x_max = torch.max(x, -2)[0]
                x_max = x_max.unsqueeze(2).repeat(1, 1, x.shape[2], 1)
                x = torch.cat([x, x_max], dim=-1)
        x_max = torch.max(x, -2)[0]
        return x_max


class ChoiceHead(nn.Module):
    def __init__(self, d_model, out_size, dropout, choices=3):
        super(ChoiceHead, self).__init__()
        self.model_list = nn.ModuleList(
            [GeneratorWithParallelHeads626(d_model, out_size, dropout) for i in range(choices)])

    def forward(self, x, idx):
        pred_coord, pred_class = [], []
        for m in self.model_list:
            res1, res2 = m(x)
            pred_coord.append(res1.unsqueeze(-1))
            pred_class.append(res2.unsqueeze(-1))
        pred_coord = torch.cat(pred_coord, dim=-1)
        pred_class = torch.cat(pred_class, dim=-1)
        idx = idx - (idx > 0).int()

        k = pred_coord.shape[2]
        pred_coord = torch.gather(pred_coord, dim=-1, index=idx.view(*idx.shape, 1, 1, 1, 1).repeat(1, 1, k, 80, 2, 1))
        pred_class = torch.gather(pred_class, dim=-1, index=idx.view(*idx.shape, 1, 1).repeat(1, 1, k, 1))
        pred_coord = pred_coord.squeeze(-1)
        pred_class = pred_class.squeeze(-1)
        return pred_coord, pred_class

class deeperChoiceHead(nn.Module):
    def __init__(self, d_model, out_size, dropout, choices=3):
        super(deeperChoiceHead, self).__init__()
        self.model_list = nn.ModuleList(
            [deeper626(d_model, out_size, dropout) for i in range(choices)])

    def forward(self, x, idx):
        pred_coord, pred_class = [], []
        for m in self.model_list:
            res1, res2 = m(x)
            pred_coord.append(res1.unsqueeze(-1))
            pred_class.append(res2.unsqueeze(-1))
        pred_coord = torch.cat(pred_coord, dim=-1)
        pred_class = torch.cat(pred_class, dim=-1)
        idx = idx - (idx > 0).int()

        k = pred_coord.shape[2]
        pred_coord = torch.gather(pred_coord, dim=-1, index=idx.view(*idx.shape, 1, 1, 1, 1).repeat(1, 1, k, 80, 2, 1))
        pred_class = torch.gather(pred_class, dim=-1, index=idx.view(*idx.shape, 1, 1).repeat(1, 1, k, 1))
        pred_coord = pred_coord.squeeze(-1)
        pred_class = pred_class.squeeze(-1)
        return pred_coord, pred_class

class intChoiceHead(nn.Module):
    def __init__(self, d_model, out_size, dropout, choices=3):
        super(intChoiceHead, self).__init__()
        self.model_list = nn.ModuleList(
            [GeneratorWithInteraction(d_model, out_size, dropout) for i in range(choices)])

    def forward(self, x, idx):
        pred_coord, pred_class = [], []
        for m in self.model_list:
            res1, res2 = m(x)
            pred_coord.append(res1.unsqueeze(-1))
            pred_class.append(res2.unsqueeze(-1))
        pred_coord = torch.cat(pred_coord, dim=-1)
        pred_class = torch.cat(pred_class, dim=-1)
        #idx = idx - (idx > 0).int()
        idx = idx[...,1]
        idx = idx - (idx > 0).int()
        k = pred_coord.shape[2]
        pred_coord = torch.gather(pred_coord, dim=-1, index=idx.view(*idx.shape,1, 1, 1, 1, 1).repeat(1,2,  k, 80, 2, 1))
        pred_class = torch.gather(pred_class, dim=-1, index=idx.view(*idx.shape, 1,1).repeat(1, k, 1))
        pred_coord = pred_coord.squeeze(-1)
        pred_class = pred_class.squeeze(-1)
        return pred_coord, pred_class



class rr_prediction_head(nn.Module):
    def __init__(self,input_dim, dropout):
        super(rr_prediction_head, self).__init__()
        self.head_veh = nn.Sequential(
                        PointerwiseFeedforward(input_dim, 2 * input_dim, dropout=dropout),
                        nn.Linear(input_dim, 128, bias=True),
                        nn.Linear(128, 1, bias=True),
                        nn.Sigmoid())
        self.head_ped = nn.Sequential(
                        PointerwiseFeedforward(input_dim, 2 * input_dim, dropout=dropout),
                        nn.Linear(input_dim, 128, bias=True),
                        nn.Linear(128, 1, bias=True),
                        nn.Sigmoid())
        self.head_cyc = nn.Sequential(
                        PointerwiseFeedforward(input_dim, 2 * input_dim, dropout=dropout),
                        nn.Linear(input_dim,128, bias=True),
                        nn.Linear(128, 1, bias=True),
                        nn.Sigmoid())

    def forward(self, x, idx):
        veh_out = self.head_veh(x).unsqueeze(-1)
        ped_out = self.head_ped(x).unsqueeze(-1)
        cyc_out = self.head_cyc(x).unsqueeze(-1)
        out = torch.cat([veh_out,ped_out,cyc_out],-1)
        idx = idx.view(*idx.shape,1,1,1).repeat(1,1,*out.shape[-3:-1],1)
        prob = torch.gather(out,-1,idx).squeeze(-1)
        return prob

class kq_prediction_head(nn.Module):
    def __init__(self,dropout):
        super( kq_prediction_head, self).__init__()
        self.head_veh = nn.Sequential(
                        nn.Linear(512, 256, bias=True),
                        nn.LayerNorm(256),
                        nn.ReLU(),
                        PointerwiseFeedforward(256, 512, dropout=dropout),
                        nn.Linear(256, 128, bias=True),
                        nn.Linear(128, 1, bias=True),
                        nn.Sigmoid())

        self.head_ped = nn.Sequential(
                        nn.Linear(512, 256, bias=True),
                        nn.LayerNorm(256),
                        nn.ReLU(),
                        PointerwiseFeedforward(256, 512, dropout=dropout),
                        nn.Linear(256, 128, bias=True),
                        nn.Linear(128, 1, bias=True),
                        nn.Sigmoid())
        self.head_cyc = nn.Sequential(
                        nn.Linear(512, 256, bias=True),
                        nn.LayerNorm(256),
                        nn.ReLU(),
                        PointerwiseFeedforward(256, 512, dropout=dropout),
                        nn.Linear(256, 128, bias=True),
                        nn.Linear(128, 1, bias=True),
                        nn.Sigmoid())

    def forward(self, x, idx):
        veh_out = self.head_veh(x).unsqueeze(-1)
        ped_out = self.head_ped(x).unsqueeze(-1)
        cyc_out = self.head_cyc(x).unsqueeze(-1)
        out = torch.cat([veh_out,ped_out,cyc_out],-1)
        idx = idx[:,1]
        idx = idx.view(idx.shape[0],1,1,1).repeat(1,out.shape[1],1,1)
        prob = torch.gather(out,-1,idx).squeeze(-1)
        return prob


class intChoiceHead1(nn.Module):
    def __init__(self, d_model, out_size, dropout, choices=3):
        super(intChoiceHead1, self).__init__()
        self.model_list = nn.ModuleList(
            [GeneratorWithInteraction1(d_model, out_size, dropout) for i in range(choices)])

    def forward(self, x, idx):
        pred_coord, pred_class = [], []
        for m in self.model_list:
            res1, res2 = m(x)
            pred_coord.append(res1.unsqueeze(-1))
            pred_class.append(res2.unsqueeze(-1))
        pred_coord = torch.cat(pred_coord, dim=-1)
        pred_class = torch.cat(pred_class, dim=-1)
        #idx = idx - (idx > 0).int()
        idx = idx[...,1]
        idx = idx - (idx > 0).int()
        k = pred_coord.shape[2]
        pred_coord = torch.gather(pred_coord, dim=-1, index=idx.view(*idx.shape,1, 1, 1, 1, 1).repeat(1,2,  k, 80, 2, 1))
        pred_class = torch.gather(pred_class, dim=-1, index=idx.view(*idx.shape, 1,1).repeat(1, k, 1))
        pred_coord = pred_coord.squeeze(-1)
        pred_class = pred_class.squeeze(-1)
        return pred_coord, pred_class

class newChoiceHead(nn.Module):
    def __init__(self, d_model, out_size, dropout, choices=3):
        super(newChoiceHead, self).__init__()
        self.model_list = nn.ModuleList(
            [GeneratorWithProbilities(d_model, out_size, dropout) for i in range(choices)])

    def forward(self, x, idx):
        pred_coord, pred_class = [], []
        for m in self.model_list:
            res1, res2 = m(x)
            pred_coord.append(res1.unsqueeze(-1))
            pred_class.append(res2.unsqueeze(-1))
        pred_coord = torch.cat(pred_coord, dim=-1)
        pred_class = torch.cat(pred_class, dim=-1)
        idx = idx - (idx > 0).int()

        k = pred_coord.shape[2]
        pred_coord = torch.gather(pred_coord, dim=-1, index=idx.view(*idx.shape, 1, 1, 1, 1).repeat(1, 1, k, 80, 2, 1))
        pred_class = torch.gather(pred_class, dim=-1, index=idx.view(*idx.shape, 1, 1).repeat(1, 1, k, 1))
        pred_coord = pred_coord.squeeze(-1)
        pred_class = pred_class.squeeze(-1)
        return pred_coord, pred_class


def split_dim(x: torch.Tensor, split_shape: tuple, dim: int):
    if dim < 0:
        dim = len(x.shape) + dim
    return x.reshape(*x.shape[:dim], *split_shape, *x.shape[dim + 1:])
